"""
apnea_cnn.py - 1D CNN for Respiratory Apnea Detection

Combines the flexible dataset/preprocessing pipeline from nn_fromscratch
with a deeper, regularized architecture. v2: single classification head
(simpler to explain) + stronger anti-overfitting measures based on
training curve diagnostics (per-block dropout, noise augmentation,
early stopping, label smoothing, fixed non-overlapping test windows).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

# ── Task dictionaries ──────────────────────────────────────────────────────

tasksDictNeck = {
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

# ── Signal preprocessing ───────────────────────────────────────────────────

def apply_lowpass_filter(data, cutoff, fs, order=4):
    """Zero-phase Butterworth low-pass filter.

    filtfilt avoids phase delay, keeping apnea event timestamps aligned
    with the filtered signal. Cutoff removes high-frequency noise above
    the physiological band of interest.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


# ── Dataset ─────────────────────────────────────────────────────────────────

class SWDataset(Dataset):
    """Sliding-window dataset labeled from apnea interval annotations.

    A window is labeled as apnea only if it covers >= 60% of the
    annotated event duration, avoiding ambiguous partial-event windows.
    Each window is Z-score normalized per channel.

    augment: if True, adds small Gaussian noise after normalization
    (train set only). This discourages the model from memorizing exact
    window values -- useful since overlapping windows are highly similar.
    """

    def __init__(self, base_folder, labels_csv, sampling_points, offset,
                 tasks, subjects, sensor_type='mask', classification_mode='multiclass',
                 augment=False, noise_std=0.05):

        self.signals = []
        self.window_indices = []
        self.sampling_points = sampling_points
        self.classification_mode = classification_mode
        self.augment = augment
        self.noise_std = noise_std

        self.apnea_labels = {}
        if os.path.exists(labels_csv):
            df = pd.read_csv(labels_csv)
            for _, row in df.iterrows():
                filename = os.path.basename(row['file_path'])
                self.apnea_labels[filename] = {
                    'start': row['start_time'],
                    'end':   row['end_time'],
                    'type':  row['apnea_type']
                }
        else:
            print(f"  [WARNING] CSV '{labels_csv}' not found. All labels default to Normal.")

        for subject in subjects:
            for task in tasks:
                if sensor_type == 'mask':
                    if task not in tasksDictMask: continue
                    dict_name     = tasksDictMask[task]
                    subfolder     = "Inline_PQ_Data"
                    channel_cols  = slice(1, 3)   # gauge + inspiratory differential pressure
                    target_points = 15000
                    fs            = 250.0
                    cutoff_hz     = 2.5            # respiratory band
                elif sensor_type == 'neck':
                    if task not in tasksDictNeck: continue
                    dict_name     = tasksDictNeck[task]
                    subfolder     = "Neck_Pulse_Oximeter_Data"
                    channel_cols  = slice(1, 9)    # 8 PPG/SpO2 channels
                    target_points = 15000
                    fs            = 250.0
                    cutoff_hz     = 5.0             # PPG has higher-frequency content
                else:
                    raise ValueError("sensor_type must be 'mask' or 'neck'")

                file_name = f"Subject{subject:d}_{dict_name}.csv"
                file_path = os.path.join(base_folder, subfolder,
                                         f"Subject{subject:d}", file_name)
                if not os.path.exists(file_path):
                    continue

                # Load & drop trailing artefacts (non-monotonic timestamps)
                data            = np.loadtxt(file_path, delimiter=',', skiprows=1)
                time_original   = data[:, 0]
                sensor_original = data[:, channel_cols]
                idx_max         = np.argmax(time_original)
                clean_time      = time_original[:idx_max + 1]
                clean_sensor    = sensor_original[:idx_max + 1, :]

                # Resample to a uniform grid
                if sensor_type == 'mask':
                    perfect_time = np.linspace(clean_time.min(), clean_time.max(), target_points)
                    interpolator = interp1d(clean_time, clean_sensor, axis=0,
                                           kind='linear', fill_value="extrapolate")
                    sensor_resampled = interpolator(perfect_time)
                elif sensor_type == 'neck':
                    mask_60s     = clean_time <= 60.0
                    clean_sensor = clean_sensor[mask_60s, :]
                    n = clean_sensor.shape[0]
                    if n > target_points:
                        sensor_resampled = clean_sensor[:target_points, :]
                    elif n < target_points:
                        padding = np.tile(clean_sensor[-1, :], (target_points - n, 1))
                        sensor_resampled = np.vstack((clean_sensor, padding))
                    else:
                        sensor_resampled = clean_sensor

                sensor_final  = apply_lowpass_filter(sensor_resampled, cutoff=cutoff_hz, fs=fs)
                data_selected = sensor_final.T  # (C, T_total)

                self.signals.append(data_selected)
                signal_idx   = len(self.signals) - 1
                total_points = data_selected.shape[1]

                lookup_key = file_name.replace("_pulse", "")
                apnea_info = self.apnea_labels.get(lookup_key, None)

                # Sliding window + overlap-based labeling
                for start_idx in range(0, total_points - sampling_points + 1, offset):
                    end_idx   = start_idx + sampling_points
                    w_start_t = start_idx / fs
                    w_end_t   = end_idx   / fs
                    label     = 0  # default: Normal

                    if apnea_info:
                        a_start = apnea_info['start']
                        a_end   = apnea_info['end']
                        overlap_start    = max(w_start_t, a_start)
                        overlap_end      = min(w_end_t,   a_end)
                        overlap_duration = max(0.0, overlap_end - overlap_start)
                        apnea_duration   = a_end - a_start

                        # Window must cover >= 60% of the event to be labeled positive
                        if apnea_duration > 0 and (overlap_duration / apnea_duration) >= 0.8:
                            if classification_mode == 'binary':
                                label = 1
                            else:
                                if '10' in str(apnea_info['type']):
                                    label = 1
                                elif '20' in str(apnea_info['type']):
                                    label = 2

                    self.window_indices.append((signal_idx, start_idx, label))

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        signal_idx, start, label = self.window_indices[idx]
        end    = start + self.sampling_points
        window = self.signals[signal_idx][:, start:end].copy()

        # Per-channel Z-score: removes DC offset / scale differences across subjects
        mean   = window.mean(axis=1, keepdims=True)
        std    = window.std(axis=1,  keepdims=True) + 1e-8
        window = (window - mean) / std

        # Gaussian jitter (train only): forces robustness instead of memorization
        if self.augment:
            window = window + np.random.normal(0, self.noise_std, window.shape)

        return (torch.tensor(window, dtype=torch.float32),
                torch.tensor(label,  dtype=torch.long))


# ── Conv block ──────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm1d -> ReLU -> Dropout1d -> MaxPool1d(2).

    BatchNorm stabilizes gradients in deeper stacks. Dropout1d zeroes
    whole feature channels (not just individual timesteps), which is the
    right granularity for conv features and directly fights overfitting.
    MaxPool keeps the strongest activations (sharp pressure changes)
    rather than uniformly downsampling like a strided conv would.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout1d(dropout),
            nn.MaxPool1d(kernel_size=2)
        )

    def forward(self, x):
        return self.block(x)


# ── Model ───────────────────────────────────────────────────────────────────

class ApneaCNN1D(nn.Module):
    """Deep 1D CNN for apnea classification (single head).

    Input: (batch, C, 7500) -- 30s window at 250 Hz. C=2 (mask) or C=8 (neck).

    4 conv blocks, channels 2/8 -> 32 -> 64 -> 128 -> 64:
      - Kernel sizes shrink with depth (15 -> 9 -> 5 -> 3) because each
        MaxPool(2) halves the temporal resolution, so a fixed real-world
        receptive field needs a smaller kernel at deeper, coarser stages.
      - Channels grow to 128 (peak representational capacity) then
        bottleneck back to 64, which cuts classifier parameters and acts
        as a mild regularizer.
      - Dropout1d probability increases with depth (0.1 -> 0.2) since
        deeper layers have more capacity and are more prone to overfit.

    AdaptiveAvgPool1d(16) turns the variable-length feature map into a
    fixed-size vector and aggregates evidence across the whole window,
    matching the task (apnea can occur anywhere in the 30s).

    Single FC head (128 -> num_classes) with Dropout(0.4): a single head
    is simpler to justify than a multi-task setup, and most of the
    earlier overfitting gap was not actually fixed by the auxiliary head,
    so it was dropped in favor of regularizing the shared trunk instead.
    """

    def __init__(self, in_channels, classification_mode='multiclass'):
        super().__init__()
        num_classes = 2 if classification_mode == 'binary' else 3

        self.block1 = ConvBlock(in_channels, 32,  kernel_size=15, padding=7, dropout=0.10)
        self.block2 = ConvBlock(32,          64,  kernel_size=9,  padding=4, dropout=0.15)
        self.block3 = ConvBlock(64,          128, kernel_size=5,  padding=2, dropout=0.20)
        self.block4 = ConvBlock(128,         64,  kernel_size=3,  padding=1, dropout=0.20)

        self.global_pool = nn.AdaptiveAvgPool1d(16)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        return self.classifier(x)


# ── Train / eval loops ─────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, device, is_train=True):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(X)
            loss = criterion(logits, y)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X.size(0)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += X.size(0)

    return (total_loss / total, correct / total) if total > 0 else (0.0, 0.0)


def test_model(model, loader, device, classification_mode):
    """Final evaluation. Accuracy alone is misleading on imbalanced data --
    check per-class recall/F1 (apnea recall especially) in the report below.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X, y   = X.to(device), y.to(device)
            preds  = model(X).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    total   = len(all_labels)
    acc     = correct / total

    print(f"\n[TEST] Accuracy: {acc:.4f}  ({correct}/{total})")
    print("-" * 60)
    print("CONFUSION MATRIX")
    print(confusion_matrix(all_labels, all_preds))
    print("-" * 60)

    target_names = (['Normal', 'Apnea']
                    if classification_mode == 'binary'
                    else ['Normal', 'Apnea 10s', 'Apnea 20s'])
    print("CLASSIFICATION REPORT")
    print(classification_report(all_labels, all_preds,
                                target_names=target_names, zero_division=0))
    print("-" * 60 + "\n")
    return acc


def plot_curves(train_losses, val_losses, train_accs, val_accs,
                save_path="training_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="train"); ax1.plot(val_losses, label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(train_accs,  label="train"); ax2.plot(val_accs,   label="val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Curves saved to '{save_path}'")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    BASE_FOLDER = "physionet.org/files/respiratory-oximetry-apnoea/1.0.0"
    LABELS_CSV  = "mask_apnea_labels.csv"

    SENSOR_TYPE         = 'neck' 
    ''        # 'mask' (2 ch) | 'neck' (8 ch)
    CLASSIFICATION_MODE = 'binary' 
    ''  # 'binary'      | 'multiclass'

    SAMPLING_POINTS = 15000   # 45s x 250 Hz

    TRAIN_OFFSET = 500
    VAL_OFFSET   = 500

    # Test stride = full window length -> truly non-overlapping windows.
    TEST_OFFSET = SAMPLING_POINTS

    INPUT_CHANNELS = 2 if SENSOR_TYPE == 'mask' else 8

    TASKS          = list(range(0, 8))
    TRAIN_SUBJECTS = list(range(1,  15))
    VAL_SUBJECTS   = list(range(15, 18))
    TEST_SUBJECTS  = list(range(18, 21))

    BATCH_SIZE  = 16
    NUM_EPOCHS  = 40
    LR          = 1e-3
    WEIGHT_DECAY = 5e-4        # strong L2 penalty
    LABEL_SMOOTHING = 0.05     # discourages overconfident predictions
    NOISE_STD   = 0.05         # train-time Gaussian jitter (data is Z-scored)
    EARLY_STOP_PATIENCE = 8    # stop if val_loss doesn't improve for N epochs

    CHECKPOINT = f"best_model_{SENSOR_TYPE}_{CLASSIFICATION_MODE}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Sensor: {SENSOR_TYPE.upper()} | Mode: {CLASSIFICATION_MODE.upper()}\n")

    print("Loading datasets and generating windows...")
    train_ds = SWDataset(BASE_FOLDER, LABELS_CSV, SAMPLING_POINTS, TRAIN_OFFSET,
                         TASKS, TRAIN_SUBJECTS, SENSOR_TYPE, CLASSIFICATION_MODE,
                         augment=True, noise_std=NOISE_STD)
    val_ds   = SWDataset(BASE_FOLDER, LABELS_CSV, SAMPLING_POINTS, VAL_OFFSET,
                         TASKS, VAL_SUBJECTS,   SENSOR_TYPE, CLASSIFICATION_MODE)
    test_ds  = SWDataset(BASE_FOLDER, LABELS_CSV, SAMPLING_POINTS, TEST_OFFSET,
                         TASKS, TEST_SUBJECTS,  SENSOR_TYPE, CLASSIFICATION_MODE)

    print(f"Windows - Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Imbalance is handled two ways: WeightedRandomSampler rebalances which
    # samples appear in each batch, and class-weighted CrossEntropyLoss
    # scales the gradient penalty for misclassifying rare classes. They act
    # at different points (batch composition vs. gradient magnitude).
    y_train = [label for _, _, label in train_ds.window_indices]
    label_counts     = Counter(y_train)
    expected_classes = 2 if CLASSIFICATION_MODE == 'binary' else 3
    print(f"Train class distribution: {dict(sorted(label_counts.items()))}")

    sample_weights = [1.0 / label_counts[label] for label in y_train]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(train_ds), replacement=True)

    classes = np.unique(y_train)
    if len(classes) < expected_classes:
        print(f"[WARNING] Only classes {classes} present. Using uniform loss weights.")
        class_weights = torch.ones(expected_classes, dtype=torch.float32).to(device)
    else:
        weights       = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        print(f"Loss class weights: {dict(zip(classes, weights.round(3)))}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)

    model = ApneaCNN1D(in_channels=INPUT_CHANNELS,
                       classification_mode=CLASSIFICATION_MODE).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}\n")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    print(f"{'Epoch':>6}  {'Tr Loss':>8}  {'Tr Acc':>7}  {'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}")
    print("-" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss,  tr_acc  = run_epoch(model, train_loader, optimizer, criterion, device, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader,   None,      criterion, device, is_train=False)

        train_losses.append(tr_loss); val_losses.append(val_loss)
        train_accs.append(tr_acc);    val_accs.append(val_acc)
        scheduler.step(val_loss)

        best_marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT)
            best_marker = " *"
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"{epoch:6d}  {tr_loss:8.4f}  {tr_acc:7.4f}  {val_loss:8.4f}  {val_acc:7.4f}  "
              f"{optimizer.param_groups[0]['lr']:.1e}{best_marker}")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping: val_loss has not improved for {EARLY_STOP_PATIENCE} epochs.")
            break

    print(f"\nLoading best checkpoint '{CHECKPOINT}'...")
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    test_model(model, test_loader, device, CLASSIFICATION_MODE)

    plot_curves(train_losses, val_losses, train_accs, val_accs,
                save_path=f"curves_{SENSOR_TYPE}_{CLASSIFICATION_MODE}.png")


if __name__ == "__main__":
    main()