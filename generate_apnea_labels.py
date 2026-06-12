"""
generate_apnea_labels.py

Scan all mask pressure files, detect the single apnea event per file using
the known duration from the filename, and save a CSV with columns:
    subject, task_name, file_path, start_time, end_time, apnea_type
"""

import os
import pandas as pd
from apnea_detector import load_and_resample_mask, detect_single_apnea, butter_lowpass_filter

BASE_FOLDER = "physionet.org/files/respiratory-oximetry-apnoea/1.0.0"
SUBJECTS = list(range(1, 21))

TASKS_MASK = {
    2: ("4cmH2O_apnea1", 10),
    3: ("4cmH2O_apnea2", 20),
    5: ("8cmH2O_apnea1", 10),
    6: ("8cmH2O_apnea2", 20),
}

# Signal Processing Parameters
TARGET_FS = 250.0  # Sampling frequency in Hz
LOWPASS_CUTOFF = 2.5 # Cutoff frequency in Hz (typically < 3Hz for breathing)
FILTER_ORDER = 4

def process_mask_files():
    results = []
    for subject in SUBJECTS:
        for task_idx, (task_name, expected_dur) in TASKS_MASK.items():
            file_name = f"Subject{subject}_{task_name}.csv"
            file_path = os.path.join(BASE_FOLDER, "Inline_PQ_Data",
                                     f"Subject{subject}", file_name)
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found")
                continue

            print(f"Processing: {file_path}")
            try:
                t, gauge, insp = load_and_resample_mask(file_path, target_fs=TARGET_FS)
                
                insp_filtered = butter_lowpass_filter(
                    insp, 
                    cutoff=LOWPASS_CUTOFF, 
                    fs=TARGET_FS, 
                    order=FILTER_ORDER
                )
                
                apnea = detect_single_apnea(t, insp_filtered, expected_duration=expected_dur, target_fs=TARGET_FS)
                
                if apnea is None:
                    print(f"  -> No apnea detected (threshold may need adjustment)")
                    continue
                
                start, end, atype = apnea
                results.append({
                    "subject": subject,
                    "task_name": task_name,
                    "file_path": file_path,
                    "start_time": start,
                    "end_time": end,
                    "apnea_type": atype
                })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return pd.DataFrame(results)

def main():
    df = process_mask_files()
    output_csv = "mask_apnea_labels.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} intervals to '{output_csv}'")
    print("\nPreview:")
    print(df.head(10))

if __name__ == "__main__":
    main()