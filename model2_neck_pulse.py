import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from pathlib import Path

tasksDict = {
    0: "Baseline",
    1: "0cmH2O_normal_pulse",
    2: "4cmH2O_normal_pulse",
    3: "4cmH2O_apnea1_pulse",
    4: "4cmH2O_apnea2_pulse",
    5: "8cmH2O_normal_pulse",
    6: "8cmH2O_apnea1_pulse",
    7: "8cmH2O_apnea2_pulse"
}

class SWDatasetNeck(Dataset):
    def __init__(self, folder_path, sampling_points, offset, tasks, subjects, freq=100):
        self.signals = []
        self.window_indices = []
        self.sampling_points = sampling_points
        base_path = Path(folder_path)

        for subject in subjects:
            for task in tasks:
                file_name = f"Subject{subject:d}_{tasksDict[task]}.csv"
                file_path = base_path / f"Subject{subject:d}" / file_name

                if not file_path.exists():
                    print(f"[WARNING]: File Not Found: {file_path}")
                    continue

                data = np.loadtxt(file_path, delimiter=',', skiprows=1)

                time_original = data[:, 0]
                # Neck pulse only has 1 signal channel of interest (column index 1)
                pressure_original = data[:, 1].flatten()

                idx_max = np.argmax(time_original)
                clean_time = time_original[:idx_max + 1]
                clean_pressure = pressure_original[:idx_max + 1]

                # Resample onto a perfect uniform grid based on the full 60s file lifecycle
                total_desired_points = 60 * freq
                perfect_time = np.linspace(clean_time.min(), clean_time.max(), total_desired_points)

                interpolator = interp1d(clean_time, clean_pressure, axis=0,
                                        kind='linear', fill_value="extrapolate")
                corrected_data = interpolator(perfect_time)

                # Shape it properly into a single channel array format: (1, time_points)
                data_selected = np.expand_dims(corrected_data, axis=0)

                self.signals.append(data_selected)
                signal_idx = len(self.signals) - 1
                total_points = data_selected.shape[1]

                match task:
                    case 0 | 1 | 2 | 5:
                        label = 0  # Normal breathing
                    case 3 | 6:
                        label = 1  # 10-second apnea
                    case 4 | 7:
                        label = 2  # 20-second apnea
                    case _:
                        label = 0

                for start in range(0, total_points - sampling_points + 1, offset):
                    self.window_indices.append((signal_idx, start, label))

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        signal_idx, start, label = self.window_indices[idx]
        end = start + self.sampling_points
        window = self.signals[signal_idx][:, start:end]
        return (torch.tensor(window, dtype=torch.float32), torch.tensor(label, dtype=torch.long))


class AdvancedCNN1D(nn.Module):
    # FIXED: Changed in_channels default to 1 to match the neck PPG track
    def __init__(self, in_channels=1, sampling_points=3000, num_classes=3):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.flatten = nn.Flatten()
        flatten_size = self._get_flatten_size(in_channels, sampling_points)

        self.fc_multi = nn.Sequential(
            nn.Linear(flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

        self.fc_binary = nn.Sequential(
            nn.Linear(flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def _get_flatten_size(self, in_channels, sampling_points):
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, sampling_points)
            x = self.pool1(self.relu1(self.bn1(self.conv1(dummy))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
            return x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        return self.fc_multi(x), self.fc_binary(x)


def run_epoch(model, loader, optimizer, criterion, device, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for X, y_multi in loader:
            X, y_multi = X.to(device), y_multi.to(device)
            y_binary = (y_multi > 0).long().to(device)

            if is_train:
                optimizer.zero_grad()

            logits_multi, logits_binary = model(X)

            loss_multi = criterion(logits_multi, y_multi)
            loss_binary = criterion(logits_binary, y_binary)
            loss = loss_multi + loss_binary

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X.size(0)
            correct += (logits_multi.argmax(1) == y_multi).sum().item()
            total += X.size(0)

    return total_loss / total, correct / total


def main():
    FOLDER_PATH = "physionet.org/files/respiratory-oximetry-apnoea/1.0.0/Neck_Pulse_Oximeter_Data"
    FREQ = 100
    WINDOW_SEC = 30
    SAMPLING_POINTS = WINDOW_SEC * FREQ  # 3000 samples

    TRAIN_OFFSET = 200
    VAL_OFFSET = 200
    TEST_OFFSET = SAMPLING_POINTS

    TASKS = list(range(1, 8))
    TRAIN_SUBJECTS = list(range(1, 15))
    VAL_SUBJECTS = list(range(15, 18))
    TEST_SUBJECTS = list(range(18, 21))

    BATCH_SIZE = 32
    NUM_EPOCHS = 40
    LR = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Use our updated custom local neck loader class
    train_ds = SWDatasetNeck(FOLDER_PATH, SAMPLING_POINTS, TRAIN_OFFSET, TASKS, TRAIN_SUBJECTS, FREQ)
    val_ds = SWDatasetNeck(FOLDER_PATH, SAMPLING_POINTS, VAL_OFFSET, TASKS, VAL_SUBJECTS, FREQ)
    test_ds = SWDatasetNeck(FOLDER_PATH, SAMPLING_POINTS, TEST_OFFSET, TASKS, TEST_SUBJECTS, FREQ)

    print(f"Windows — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Instantiate model with 1 input channel
    model = AdvancedCNN1D(in_channels=1, sampling_points=SAMPLING_POINTS, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, device, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader, None, criterion, device, is_train=False)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:2d}/{NUM_EPOCHS}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    # Final Evaluation Cycle
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y_multi in test_loader:
            X, y_multi = X.to(device), y_multi.to(device)
            logits_multi, _ = model(X)
            preds = logits_multi.argmax(1)
            correct += (preds == y_multi).sum().item()
            total += X.size(0)

    acc = correct / total
    print(f"[TEST] Accuracy: {acc:.4f}  ({correct}/{total})")

    # Generate Training Analytics Charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="train")
    ax1.plot(val_losses, label="val")
    ax1.set_title("Loss");
    ax1.set_xlabel("Epoch");
    ax1.legend()

    ax2.plot(train_accs, label="train")
    ax2.plot(val_accs, label="val")
    ax2.set_title("Accuracy");
    ax2.set_xlabel("Epoch");
    ax2.legend()

    plt.tight_layout()
    plt.savefig("model2_neck_pulse_curves.png")
    plt.show()


if __name__ == "__main__":
    main()