"""
visualize_apnea.py

For each mask file that contains an apnea event (task names ending with 'apnea1' or 'apnea2'),
load the signal, detect the single apnea interval, and plot the result with shaded region.

Usage:
    python visualize_apnea.py                   # shows plots one by one (press any key to continue)
    python visualize_apnea.py --save            # saves plots as PNG files in './apnea_plots/'
    python visualize_apnea.py --subject 5      # only process a specific subject
"""

import os
import argparse
import matplotlib.pyplot as plt
from apnea_detector import load_and_resample_mask, detect_single_apnea

# Configuration
BASE_FOLDER = "physionet.org/files/respiratory-oximetry-apnoea/1.0.0"
SUBJECTS = list(range(1, 21))
TASKS_MASK = {
    2: "4cmH2O_apnea1",
    3: "4cmH2O_apnea2",
    5: "8cmH2O_apnea1",
    6: "8cmH2O_apnea2",
}
DETECTION_PARAMS = {
    'std_threshold': 0.03,
    'min_duration': 4.0,
    'window_sec': 1.0,
    'target_fs': 250.0
}

def plot_apnea_detection(file_path, save=False, output_dir="apnea_plots"):
    """
    Load a mask file, detect the single apnea interval, and plot the inspiratory signal
    with shaded apnea region.
    """
    # Load and resample
    t, _, insp = load_and_resample_mask(file_path)
    # Determine expected duration from filename
    expected_dur = 10 if "apnea1" in file_path else 20
    apnea = detect_single_apnea(t, insp, expected_duration=expected_dur,
                                
                                window_sec=DETECTION_PARAMS['window_sec'],
                                target_fs=DETECTION_PARAMS['target_fs'])

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(t, insp, 'b-', linewidth=0.8, label='Inspiratory ΔP')

    if apnea is not None:
        start, end, atype = apnea
        colors = {'10s apnea': 'orange', '20s apnea': 'red'}
        color = colors.get(atype, 'purple')
        ax.axvspan(start, end, alpha=0.3, color=color, label=atype)
        # Add vertical lines at start/end for clarity
        ax.axvline(start, color=color, linestyle='--', linewidth=0.8)
        ax.axvline(end, color=color, linestyle='--', linewidth=0.8)
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, 'No apnea detected', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='red')
        ax.set_title(f"{os.path.basename(file_path)}\nDetection failed")

    # Title and labels
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Inspiratory Pressure [cmH₂O]")
    if apnea:
        ax.set_title(f"{os.path.basename(file_path)}\nDetected {atype} ({start:.2f}–{end:.2f} s)")
    ax.grid(True, alpha=0.3)

    if save:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_path = os.path.join(output_dir, f"{base_name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Visualize apnea detection on mask files.")
    parser.add_argument("--save", action="store_true", help="Save plots to disk instead of showing.")
    parser.add_argument("--subject", type=int, help="Only process this subject number.")
    parser.add_argument("--output_dir", default="apnea_plots", help="Directory to save plots (if --save).")
    args = parser.parse_args()

    subjects = [args.subject] if args.subject else SUBJECTS

    for subject in subjects:
        for task_name in TASKS_MASK.values():
            file_name = f"Subject{subject}_{task_name}.csv"
            file_path = os.path.join(BASE_FOLDER, "Inline_PQ_Data",
                                     f"Subject{subject}", file_name)
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            print(f"Processing: {file_path}")
            plot_apnea_detection(file_path, save=args.save, output_dir=args.output_dir)

            if not args.save:
                input("Press Enter to continue to the next file...")

if __name__ == "__main__":
    main()