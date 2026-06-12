import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────────────────────────────────────
# DICTIONARIES
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL PROCESSING UTILS
# ─────────────────────────────────────────────────────────────────────────────

def apply_lowpass_filter(data, cutoff, fs, order=4):
    """Zero-phase low-pass filter to smooth the signal before windowing."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SWDataset(Dataset):
    """
    Sliding-window dataset that labels windows based on exact apnea intervals
    provided by an external CSV file.
    """
    def __init__(self, base_folder, labels_csv, sampling_points, offset,
                 tasks, subjects, sensor_type='mask'):

        self.signals = []
        self.window_indices = []
        self.sampling_points = sampling_points

        # Load apnea labels into a dictionary keyed by filename
        self.apnea_labels = {}
        if os.path.exists(labels_csv):
            df = pd.read_csv(labels_csv)
            for _, row in df.iterrows():
                filename = os.path.basename(row['file_path'])
                self.apnea_labels[filename] = {
                    'start': row['start_time'],
                    'end': row['end_time'],
                    'type': row['apnea_type']
                }
        else:
            print(f"  [WARNING] Labels CSV '{labels_csv}' not found. Defaulting all to normal.")

        for subject in subjects:
            for task in tasks:
                if sensor_type == 'mask':
                    if task not in tasksDictMask: continue
                    dict_name     = tasksDictMask[task]
                    subfolder     = "Inline_PQ_Data"
                    channel_cols  = slice(1, 3) 
                    target_points = 15000       
                    fs = 250.0                  
                    cutoff_hz = 2.5
                elif sensor_type == 'neck':
                    if task not in tasksDictNeck: continue
                    dict_name     = tasksDictNeck[task]
                    subfolder     = "Neck_Pulse_Oximeter_Data"
                    channel_cols  = slice(1, 9) 
                    target_points = 15000       
                    fs = 250.0
                    cutoff_hz = 5.0 
                else:
                    raise ValueError("sensor_type must be 'mask' or 'neck'")

                file_name = f"Subject{subject:d}_{dict_name}.csv"
                file_path = os.path.join(base_folder, subfolder, f"Subject{subject:d}", file_name)

                if not os.path.exists(file_path):
                    continue

                # Load & remove tail artefacts
                data            = np.loadtxt(file_path, delimiter=',', skiprows=1)
                time_original   = data[:, 0]
                sensor_original = data[:, channel_cols]

                idx_max      = np.argmax(time_original)
                clean_time   = time_original[:idx_max + 1]
                clean_sensor = sensor_original[:idx_max + 1, :]

                # 1. Resample to uniform grid
                if sensor_type == 'mask':
                    perfect_time = np.linspace(clean_time.min(), clean_time.max(), target_points)
                    interpolator = interp1d(clean_time, clean_sensor, axis=0, kind='linear', fill_value="extrapolate")
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

                # 2. Filter the signal
                sensor_final = apply_lowpass_filter(sensor_resampled, cutoff=cutoff_hz, fs=fs)
                data_selected = sensor_final.T  # (C, T_total)
                
                self.signals.append(data_selected)
                signal_idx   = len(self.signals) - 1
                total_points = data_selected.shape[1]

                lookup_key = file_name.replace("_pulse", "")
                apnea_info = self.apnea_labels.get(lookup_key, None)

                for start_idx in range(0, total_points - sampling_points + 1, offset):
                    end_idx = start_idx + sampling_points
                    
                    w_start_t = start_idx / fs
                    w_end_t = end_idx / fs
                    label = 0  # Default to Normal

                    if apnea_info:
                        a_start = apnea_info['start']
                        a_end = apnea_info['end']
                        
                        # Calculate time overlap between window and apnea event
                        overlap_start = max(w_start_t, a_start)
                        overlap_end = min(w_end_t, a_end)
                        overlap_duration = max(0, overlap_end - overlap_start)
                        window_duration = w_end_t - w_start_t
                        
                        # If >= 50% of the window contains the apnea event, assign the label
                        if (overlap_duration / window_duration) >= 0.5:
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
        
        # Z-score normalization per window
        mean   = window.mean(axis=1, keepdims=True)
        std    = window.std(axis=1,  keepdims=True) + 1e-8
        window = (window - mean) / std

        return (torch.tensor(window, dtype=torch.float32),
                torch.tensor(label,  dtype=torch.long))

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
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
            ConvBlock(in_channels, 16, kernel_size=11),
            ConvBlock(16,          32, kernel_size=7), 
        )
        self.global_pool = nn.AdaptiveAvgPool1d(8)
        self.classifier = nn.Sequential(
            nn.Flatten(),                          
            nn.Linear(32 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOPS
# ─────────────────────────────────────────────────────────────────────────────

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

    if total == 0: return 0, 0
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

    if total == 0: return 0, 0
    return total_loss / total, correct / total

def test(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y   = X.to(device), y.to(device)
            preds  = model(X).argmax(1)
            correct += (preds == y).sum().item()
            total   += X.size(0)

    if total == 0: return 0
    acc = correct / total
    print(f"\n[TEST] Final Accuracy: {acc:.4f}  ({correct}/{total})\n")
    return acc

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    BASE_FOLDER = "physionet.org/files/respiratory-oximetry-apnoea/1.0.0"
    LABELS_CSV  = "mask_apnea_labels.csv" 
    
    SENSOR_TYPE = 'mask'   # Change this to 'neck' and it will now successfully grab the mask labels!

    SAMPLING_POINTS = 7500  # 30 seconds of data per window at 250 Hz
    TRAIN_OFFSET    = 250   # Slide window by 1 second
    VAL_OFFSET      = 250
    TEST_OFFSET     = 2500  # No overlap for testing
    
    INPUT_CHANNELS  = 2 if SENSOR_TYPE == 'mask' else 8

    TASKS          = list(range(0, 8))
    TRAIN_SUBJECTS = list(range(1,  15))   
    VAL_SUBJECTS   = list(range(15, 18))   
    TEST_SUBJECTS  = list(range(18, 21))   

    BATCH_SIZE  = 16 
    NUM_EPOCHS  = 20         
    LR          = 1e-3
    
    CHECKPOINT = f"best_model_{SENSOR_TYPE}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Sensor: {SENSOR_TYPE.upper()}\n")

    print("Loading datasets and generating localized windows...")
    train_ds = SWDataset(BASE_FOLDER, LABELS_CSV, SAMPLING_POINTS, TRAIN_OFFSET, TASKS, TRAIN_SUBJECTS, SENSOR_TYPE)
    val_ds   = SWDataset(BASE_FOLDER, LABELS_CSV, SAMPLING_POINTS, VAL_OFFSET, TASKS, VAL_SUBJECTS, SENSOR_TYPE)
    test_ds  = SWDataset(BASE_FOLDER, LABELS_CSV, SAMPLING_POINTS, TEST_OFFSET, TASKS, TEST_SUBJECTS, SENSOR_TYPE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Compute robust class weights with a Failsafe
    # ─────────────────────────────────────────────────────────────────────────
    y_train = [label for _, _, label in train_ds.window_indices]
    classes = np.unique(y_train)
    
    if len(classes) < 3:
        print(f"\n[WARNING] Only found classes {classes} in training data.")
        print("Defaulting to unweighted loss (1.0 for all) to prevent crash.")
        class_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(device)
    else:
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        print(f"Computed Class Weights: {weights}")

    model = CNN1D(in_channels=INPUT_CHANNELS, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    print(f"\n{'Epoch':>6}  {'Tr Loss':>8}  {'Tr Acc':>7}  {'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}")
    print("─" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss,  tr_acc  = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(tr_loss);  val_losses.append(val_loss)
        train_accs.append(tr_acc);     val_accs.append(val_acc)
        scheduler.step(val_loss)

        best_marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT)
            best_marker = " *"

        print(f"{epoch:6d}  {tr_loss:8.4f}  {tr_acc:7.4f}  {val_loss:8.4f}  {val_acc:7.4f}  {optimizer.param_groups[0]['lr']:.1e}{best_marker}")

    print(f"\nLoading best checkpoint '{CHECKPOINT}'...")
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    test(model, test_loader, device)

if __name__ == "__main__":
    main()