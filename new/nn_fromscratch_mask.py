import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d

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


class SWDataset(Dataset):
    # sampling_points : window size * frequency
    # offset          : stride between windows
    # tasks           : list of task indices to load
    # subjects        : list of subject indices to load (enables train/val/test split)
    def __init__(self, folder_path, sampling_points, offset, tasks, subjects, freq=100):
        self.signals = []
        self.window_indices = []
        self.sampling_points = sampling_points
 
        for subject in subjects:
            for task in tasks:
                file_name = f"Subject{subject:d}_{tasksDict[task]}.csv"
                file_path = os.path.join(folder_path, f"Subject{subject:d}", file_name)
 
                if not os.path.exists(file_path):
                    print(f"[WARNING]: File Not Found: {file_path}")
                    continue
 
                data = np.loadtxt(file_path, delimiter=',', skiprows=1)
 
                time_original = data[:, 0]
                pressure_original = data[:, 1:3]  # Gauge + Inspiratory differential
 
                idx_max = np.argmax(time_original)
                clean_time = time_original[:idx_max + 1]
                clean_pressure = pressure_original[:idx_max + 1, :]
 
                # Resample to a perfectly uniform 60-second grid
                total_desired_points = 60 * freq
                perfect_time = np.linspace(clean_time.min(), clean_time.max(),
                                           total_desired_points)
 
                interpolator = interp1d(clean_time, clean_pressure, axis=0,
                                        kind='linear', fill_value="extrapolate")
                corrected_data = interpolator(perfect_time)
 
                data_selected = corrected_data.T          #(channels, time)
                self.signals.append(data_selected)
                signal_idx = len(self.signals) - 1
                total_points = data_selected.shape[1]
 
                match task:
                    case 0 | 1 | 2 | 5:
                        label = 0   # normal breathing
                    case 3 | 6:
                        label = 1   # apnea 10 s
                    case 4 | 7:
                        label = 2   # apnea 20 s
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
        return (torch.tensor(window, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long))




class CNN1D (torch.nn.Module):
    # in_channels (for pressure data):Gauge Pressure [cmH2O],Inspiratory differential pressure [cmH2O]

    def __init__(self, in_channels,sampling_points,num_classes=3):
        super().__init__()
        self.layer1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.layer2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.flatten = nn.Flatten()

        flatten_size = self._get_flatten_size(in_channels, sampling_points)

        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(64, num_classes)
        )

    def _get_flatten_size(self, in_channels, sampling_points):
        """Run a dummy forward pass to determine the flattened feature size."""
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, sampling_points)
            x = self.pool1(self.act1(self.layer1(dummy)))
            x = self.pool2(self.act2(self.layer2(x)))
            return x.view(1, -1).shape[1]


    def forward(self,x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.layer2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        out = self.classifier(x)

        return out
    
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
    """Evaluate on the test set and print accuracy."""
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
 
    with torch.no_grad():
        for X, y in loader:
            X, y  = X.to(device), y.to(device)
            preds  = model(X).argmax(1)
            correct += (preds == y).sum().item()
            total   += X.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
 
    acc = correct / total
    print(f"[TEST] Accuracy: {acc:.4f}  ({correct}/{total})")
    return acc, all_preds, all_labels


def main():

    FOLDER_PATH = "physionet.org/files/respiratory-oximetry-apnoea/1.0.0/Inline_PQ_Data"          # root folder of the dataset
    FREQ        = 100             # Hz (resampling target)
    WINDOW_SEC  = 30              # window length in seconds
    SAMPLING_POINTS = WINDOW_SEC * FREQ   # 2 000 samples

    # Split config
    # Subjects are split by ID so there is zero overlap between sets.
    # Test set uses a non-overlapping stride (stride == window) to avoid
    # evaluating on windows that share data with each other.

    TRAIN_OFFSET = 200            # 2 s stride = many overlapping windows
    VAL_OFFSET   = 200
    TEST_OFFSET  = SAMPLING_POINTS        # 20 s stride = non-overlapping
    
    TASKS          = list(range(1, 8))    # tasks 1-7 (skip Baseline)
    TRAIN_SUBJECTS = list(range(1,  15))  # 14 subjects
    VAL_SUBJECTS   = list(range(15, 18))  #  3 subjects
    TEST_SUBJECTS  = list(range(18, 21))  #  3 subjects

    #Training Config
    BATCH_SIZE = 32
    NUM_EPOCHS = 40
    LR         = 1e-3
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    #Datasets and DataLoaders
    train_ds = SWDataset(FOLDER_PATH, SAMPLING_POINTS, TRAIN_OFFSET,
                         TASKS, TRAIN_SUBJECTS, FREQ)
    val_ds   = SWDataset(FOLDER_PATH, SAMPLING_POINTS, VAL_OFFSET,
                         TASKS, VAL_SUBJECTS,   FREQ)
    test_ds  = SWDataset(FOLDER_PATH, SAMPLING_POINTS, TEST_OFFSET,
                         TASKS, TEST_SUBJECTS,  FREQ)
 
    print(f"Windows — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    #Model
    model = CNN1D(in_channels=2, sampling_points=SAMPLING_POINTS, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    #Training
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
 
    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss,  tr_acc  = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
 
        train_losses.append(tr_loss);  val_losses.append(val_loss)
        train_accs.append(tr_acc);     val_accs.append(val_acc)
 
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
    
    #Testing
    test(model, test_loader, device)

    #Plotting Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="train")
    ax1.plot(val_losses,   label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
 
    ax2.plot(train_accs, label="train")
    ax2.plot(val_accs,   label="val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
 
    plt.tight_layout()
    plt.savefig("training_curves_mask.png")
    plt.show()
 
 
if __name__ == "__main__":
    main()

