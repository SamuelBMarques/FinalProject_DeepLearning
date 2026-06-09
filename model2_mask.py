import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# import
from nn_fromscratch_mask import SWDataset

tasksDict = {
    0 : "Baseline",
    1 : "0cmH2O_normal",
    2 : "4cmH2O_normal",
    3 : "4cmH2O_apnea1",
    4 : "4cmH2O_apnea2",
    5 : "8cmH2O_normal",
    6 : "8cmH2O_apnea1",
    7 : "8cmH2O_apnea2"
}

class AdvancedCNN1D(nn.Module):
    # in_channels (for pressure data):Gauge Pressure [cmH2O],Inspiratory differential pressure [cmH2O]

    def __init__(self, in_channels=2, sampling_points=3000, num_classes=3):
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
        """Run a dummy forward pass to determine the flattened feature size."""
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

        logits_multi = self.fc_multi(x)
        logits_binary = self.fc_binary(x)
        return logits_multi, logits_binary


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
    FOLDER_PATH = "physionet.org/files/respiratory-oximetry-apnoea/1.0.0/Inline_PQ_Data"  # root folder of the dataset
    FREQ = 100  # Hz (resampling target)
    WINDOW_SEC = 30  # window length in seconds
    SAMPLING_POINTS = WINDOW_SEC * FREQ  # 3 000 samples

    # Split config
    # Subjects are split by ID so there is zero overlap between sets.
    # Test set uses a non-overlapping stride (stride == window) to avoid
    # evaluating on windows that share data with each other.

    TRAIN_OFFSET = 200  # 2 s stride = many overlapping windows
    VAL_OFFSET = 200
    TEST_OFFSET = SAMPLING_POINTS  # 30 s stride = non-overlapping

    TASKS = list(range(1, 8))  # tasks 1-7 (skip Baseline)
    TRAIN_SUBJECTS = list(range(1, 15))  # 14 subjects
    VAL_SUBJECTS = list(range(15, 18))  # 3 subjects
    TEST_SUBJECTS = list(range(18, 21))  # 3 subjects

    # Training Config
    BATCH_SIZE = 32
    NUM_EPOCHS = 40
    LR = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Datasets and DataLoaders
    train_ds = SWDataset(FOLDER_PATH, SAMPLING_POINTS, TRAIN_OFFSET, TASKS, TRAIN_SUBJECTS, FREQ)
    val_ds = SWDataset(FOLDER_PATH, SAMPLING_POINTS, VAL_OFFSET, TASKS, VAL_SUBJECTS, FREQ)
    test_ds = SWDataset(FOLDER_PATH, SAMPLING_POINTS, TEST_OFFSET, TASKS, TEST_SUBJECTS, FREQ)

    print(f"Windows — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = AdvancedCNN1D(in_channels=2, sampling_points=SAMPLING_POINTS, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # Training
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, device, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader, None, criterion, device, is_train=False)

        train_losses.append(tr_loss);
        val_losses.append(val_loss)
        train_accs.append(tr_acc);
        val_accs.append(val_acc)

        print(f"Epoch {epoch:2d}/{NUM_EPOCHS}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    # Testing
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

    # Plotting Training Curves
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
    plt.savefig("model2_mask_curves.png")
    plt.show()


if __name__ == "__main__":
    main()