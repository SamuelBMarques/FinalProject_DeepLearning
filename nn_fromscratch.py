import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d

# ─────────────────────────────────────────────────────────────────────────────
# DICTIONARIES: task index → filename suffix
# ─────────────────────────────────────────────────────────────────────────────

tasksDictNeck = {
    0: "Baseline",
    1: "0cmH2O_normal_pulse",
    2: "4cmH2O_normal_pulse",
    3: "4cmH2O_apnea1_pulse",
    4: "4cmH2O_apnea2_pulse",
    5: "8cmH2O_normal_pulse",
    6: "8cmH2O_apnea1_pulse",
    7: "8cmH2O_apnea2_pulse",
}

tasksDictMask = {
    0: "0cmH2O_normal",
    1: "4cmH2O_normal",
    2: "4cmH2O_apnea1",
    3: "4cmH2O_apnea2",
    4: "8cmH2O_normal",
    5: "8cmH2O_apnea1",
    6: "8cmH2O_apnea2",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SWDataset(Dataset):
    """
    Sliding-window dataset for 1-D physiological signals.
    """

    def __init__(self, base_folder, sampling_points, offset,
                 tasks, subjects, sensor_type='mask'):

        self.signals       = []
        self.window_indices = []   # (signal_idx, start_sample, label)
        self.sampling_points = sampling_points

        for subject in subjects:
            for task in tasks:

                #Sensor-specific configuration
                if sensor_type == 'mask':
                    if task not in tasksDictMask:
                        continue
                    dict_name     = tasksDictMask[task]
                    subfolder     = "Inline_PQ_Data"
                    channel_cols  = slice(1, 3)   # Gauge + Inspiratory pressure
                    target_points = 6000           # 60 s × 100 Hz

                elif sensor_type == 'neck':
                    if task not in tasksDictNeck:
                        continue
                    dict_name     = tasksDictNeck[task]
                    subfolder     = "Neck_Pulse_Oximeter_Data"
                    channel_cols  = slice(1, 9)   # 4 × 660 nm + 4 × 940 nm
                    target_points = 15000          # 60 s × 250 Hz

                else:
                    raise ValueError("sensor_type must be 'mask' or 'neck'")

                #File path
                file_name = f"Subject{subject:d}_{dict_name}.csv"
                file_path = os.path.join(
                    base_folder, subfolder, f"Subject{subject:d}", file_name)

                if not os.path.exists(file_path):
                    print(f"  [WARNING] File not found: {file_path}")
                    continue

                #Load & remove tail artefacts
                data            = np.loadtxt(file_path, delimiter=',', skiprows=1)
                time_original   = data[:, 0]
                sensor_original = data[:, channel_cols]

                idx_max      = np.argmax(time_original)
                clean_time   = time_original[:idx_max + 1]
                clean_sensor = sensor_original[:idx_max + 1, :]

                # Sensor-specific resampling 
                if sensor_type == 'mask':
                    # Mask timestamps have jitter; interpolate to perfect 100 Hz.
                    perfect_time = np.linspace(
                        clean_time.min(), clean_time.max(), target_points)
                    interpolator = interp1d(
                        clean_time, clean_sensor, axis=0,
                        kind='linear', fill_value="extrapolate")
                    sensor_final = interpolator(perfect_time)

                elif sensor_type == 'neck':
                    # Neck is already at 250 Hz; just enforce strict 60-s cutoff.
                    mask_60s     = clean_time <= 60.0
                    clean_sensor = clean_sensor[mask_60s, :]
                    n            = clean_sensor.shape[0]
                    if n > target_points:
                        sensor_final = clean_sensor[:target_points, :]
                    elif n < target_points:
                        padding      = np.tile(clean_sensor[-1, :],
                                               (target_points - n, 1))
                        sensor_final = np.vstack((clean_sensor, padding))
                    else:
                        sensor_final = clean_sensor

                data_selected = sensor_final.T            # (C, T_total)
                self.signals.append(data_selected)
                signal_idx   = len(self.signals) - 1
                total_points = data_selected.shape[1]

                # Label 
                if "normal" in dict_name or "Baseline" in dict_name:
                    label = 0
                elif "apnea1" in dict_name or "apnoea1" in dict_name:
                    label = 1   # 10-s apnea
                elif "apnea2" in dict_name or "apnoea2" in dict_name:
                    label = 2   # 20-s apnea
                else:
                    label = 0

                for start in range(0, total_points - sampling_points + 1, offset):
                    self.window_indices.append((signal_idx, start, label))

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        signal_idx, start, label = self.window_indices[idx]
        end    = start + self.sampling_points
        window = self.signals[signal_idx][:, start:end].copy()  # (C, T)
        mean   = window.mean(axis=1, keepdims=True)
        std    = window.std(axis=1,  keepdims=True) + 1e-8
        window = (window - mean) / std

        return (torch.tensor(window, dtype=torch.float32),
                torch.tensor(label,  dtype=torch.long))

class ConvBlock(nn.Module):
    """
    Conv1d → BatchNorm1d → ReLU → MaxPool1d

    BatchNorm normalises activations between layers, reducing sensitivity
    to weight initialisation and allowing higher learning rates.  It also
    acts as a mild regulariser, so we can afford a smaller Dropout rate.
    'same' padding (padding = kernel_size // 2 for odd kernels) preserves
    the temporal dimension through the conv layer; only MaxPool reduces it.
    """

    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2),   # 'same' padding
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
        )

    def forward(self, x):
        return self.block(x)


class CNN1D(nn.Module):

    def __init__(self, in_channels, num_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(in_channels, 16, kernel_size=11), # Block 1: Focuses on local waveform shapes
            ConvBlock(16,          32, kernel_size=7),  # Block 2: Captures multi-second rhythmic patterns
        )

        self.global_pool = nn.AdaptiveAvgPool1d(8)

        self.classifier = nn.Sequential(
            nn.Flatten(),                          
            nn.Linear(32 * 8, 64),                      # Reduced hidden layer from 128 to 64
            nn.ReLU(),
            nn.Dropout(0.4),                            # Increased slightly to combat overfitting
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x)


def compute_class_weights(dataset, num_classes, device):
    """
    Returns inverse-frequency class weights for CrossEntropyLoss.

    Compensates for class imbalance — the NECK sensor has twice as many
    'normal' tasks (Baseline + 3 normal_pulse) as each apnea class, which
    caused the model in v1 to collapse to always predicting class 0 and
    achieve exactly 50 % accuracy.

    Weights are scaled so their mean equals 1, keeping the effective
    gradient magnitude comparable to the unweighted case.
    """
    counts = torch.zeros(num_classes)
    for (_, _, label) in dataset.window_indices:
        counts[label] += 1

    print(f"  Class counts  : {counts.int().tolist()}")
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.mean()   # mean weight ≈ 1.0
    print(f"  Class weights : {[f'{w:.3f}' for w in weights.tolist()]}")
    return weights.to(device)


class EarlyStopping:
    """
    Stops training when validation loss stops improving.

    patience  : epochs to wait after last improvement before stopping.
    min_delta : minimum decrease to count as an improvement.

    Rationale: without early stopping the model in v1 kept training after
    its best checkpoint (epoch 33, val_loss 0.8293), then degraded after
    the loss spike at epoch 34.  Early stopping freezes training at the
    right moment and, combined with best-model checkpointing, ensures the
    test set is always evaluated on the best weights seen during training.
    """

    def __init__(self, patience=10, min_delta=1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.counter     = 0
        self.best_loss   = float('inf')
        self.should_stop = False

    def step(self, val_loss) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += X.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out  = model(X)
            loss = criterion(out, y)

            total_loss += loss.item() * X.size(0)
            correct    += (out.argmax(1) == y).sum().item()
            total      += X.size(0)

    return total_loss / total, correct / total


def test(model, loader, device):
    """Evaluates on the held-out test set using the best saved checkpoint."""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for X, y in loader:
            X, y   = X.to(device), y.to(device)
            preds  = model(X).argmax(1)
            correct += (preds == y).sum().item()
            total   += X.size(0)

    acc = correct / total
    print(f"\n======================================")
    print(f"[TEST] Final Accuracy: {acc:.4f}  ({correct}/{total})")
    print(f"======================================\n")
    return acc


def main():

    BASE_FOLDER = "physionet.org/files/respiratory-oximetry-apnoea/1.0.0"

    #Toggle sensor
    SENSOR_TYPE = 'neck'   # 'mask' (pressure, 100 Hz) | 'neck' (oximetry, 250 Hz)

    # 10-s windows 
    #   MASK : 1000 pts = 10 s × 100 Hz
    #   NECK : 2500 pts = 10 s × 250 Hz
    # A 5-s stride gives floor((60 - 10) / 5) + 1 = 11 windows per file

    SAMPLING_POINTS = 6000 if SENSOR_TYPE == 'mask' else 15000
    INPUT_CHANNELS  = 2    if SENSOR_TYPE == 'mask' else 8

    TRAIN_OFFSET = 500              # 5-s stride (50 % overlap during training)
    VAL_OFFSET   = 500
    TEST_OFFSET  = SAMPLING_POINTS  # non-overlapping test windows

    TASKS          = list(range(0, 8))
    TRAIN_SUBJECTS = list(range(1,  15))   # 14 subjects for training
    VAL_SUBJECTS   = list(range(15, 18))   #  3 subjects for validation
    TEST_SUBJECTS  = list(range(18, 21))   #  3 subjects for testing

    BATCH_SIZE  = 32
    NUM_EPOCHS  = 80          # upper bound; early stopping usually triggers earlier
    LR          = 1e-3
    ES_PATIENCE = 12          # early stopping patience (epochs)
    LR_PATIENCE = 5           # ReduceLROnPlateau patience (epochs)
    LR_FACTOR   = 0.5         # LR is halved when plateau is detected

    CHECKPOINT = f"best_model_{SENSOR_TYPE}.pt"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Sensor: {SENSOR_TYPE.upper()}\n")

    # Datasets & loaders
    print("Loading datasets...")
    train_ds = SWDataset(BASE_FOLDER, SAMPLING_POINTS, TRAIN_OFFSET,
                         TASKS, TRAIN_SUBJECTS, SENSOR_TYPE)
    val_ds   = SWDataset(BASE_FOLDER, SAMPLING_POINTS, VAL_OFFSET,
                         TASKS, VAL_SUBJECTS,   SENSOR_TYPE)
    test_ds  = SWDataset(BASE_FOLDER, SAMPLING_POINTS, TEST_OFFSET,
                         TASKS, TEST_SUBJECTS,  SENSOR_TYPE)

    print(f"Windows — Train: {len(train_ds)} | Val: {len(val_ds)} | "
          f"Test: {len(test_ds)}\n")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # Model 
    model = CNN1D(in_channels=INPUT_CHANNELS, num_classes=3).to(device)

    print("Computing class weights from training set...")
    class_weights = compute_class_weights(train_ds, num_classes=3, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=LR_PATIENCE,
        factor=LR_FACTOR)

    early_stop    = EarlyStopping(patience=ES_PATIENCE)
    best_val_loss = float('inf')

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    #Training loop
    header = (f"{'Epoch':>6}  {'Tr Loss':>8}  {'Tr Acc':>7}  "
              f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}")
    print(f"\n{header}")
    print("─" * len(header))

    for epoch in range(1, NUM_EPOCHS + 1):

        tr_loss,  tr_acc  = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(tr_loss);  val_losses.append(val_loss)
        train_accs.append(tr_acc);     val_accs.append(val_acc)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        best_marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT)
            best_marker = " *"

        print(f"{epoch:6d}  {tr_loss:8.4f}  {tr_acc:7.4f}  "
              f"{val_loss:8.4f}  {val_acc:7.4f}  {current_lr:.1e}{best_marker}")

        if early_stop.step(val_loss):
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {ES_PATIENCE} epochs).")
            break

    # Test 
    print(f"\nLoading best checkpoint '{CHECKPOINT}' "
          f"(val_loss = {best_val_loss:.4f})...")
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    test(model, test_loader, device)

    # Training curves 
    epochs_ran = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs_ran, train_losses, label="Train")
    ax1.plot(epochs_ran, val_losses,   label="Validation", linestyle='--')
    ax1.set_title("Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs_ran, train_accs, label="Train")
    ax2.plot(epochs_ran, val_accs,   label="Validation", linestyle='--')
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(0, 1)
    ax2.legend()

    plt.suptitle(
        f"Sensor: {SENSOR_TYPE.upper()} | "
        f"Window: {SAMPLING_POINTS} pts | "
        f"Best Val Loss: {best_val_loss:.4f}"
    )
    plt.tight_layout()
    out_name = f"training_curves_{SENSOR_TYPE}.png"
    plt.savefig(out_name, dpi=150)
    print(f"Curves saved as '{out_name}'.")
    plt.show()


if __name__ == "__main__":
    main()