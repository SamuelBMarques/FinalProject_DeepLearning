import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_oximeter_file(file_path, output_dir="plots"):
    """
    Reads an oximeter CSV file and generates a single image with 8 subplots 
    for each PD channel, saving it to the output directory.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load the data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # 2. Define the channels to plot
    time_col = "Time [s]"
    channels = ["PD1", "PD2", "PD3", "PD4", "PD1_9", "PD2_9", "PD3_9", "PD4_9"]
    
    # Ensure all required columns are actually in the CSV
    if time_col not in df.columns or not all(c in df.columns for c in channels):
        print(f"Skipping {file_path}: Missing required columns.")
        return

    time_data = df[time_col]

    # 3. Create a figure with 8 subplots stacked vertically
    # sharex=True ensures they all share the same time axis
    fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(12, 16), sharex=True)
    
    # Extract filename for the title
    file_name = os.path.basename(file_path)
    fig.suptitle(f"Oximeter Data: {file_name}", fontsize=16, y=0.98)

    # 4. Loop through the channels and plot them on their respective axes
    for i, (ax, channel) in enumerate(zip(axes, channels)):
        ax.plot(time_data, df[channel], color='tab:blue', linewidth=1.0)
        
        # Formatting for readability
        ax.set_ylabel(channel, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Remove top and right spines for a cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Label the bottom X-axis only
    axes[-1].set_xlabel("Time [s]", fontsize=14, fontweight='bold')

    # 5. Clean up layout and save
    plt.tight_layout()
    # Adjust layout so the suptitle doesn't overlap with the top plot
    fig.subplots_adjust(top=0.95) 
    
    # Save the figure
    output_filename = os.path.splitext(file_name)[0] + "_visualization.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig) # Close the figure to free up memory
    
    print(f"Saved plot: {output_path}")

def process_all_files(input_folder):
    """
    Scans the target folder for CSV files and processes them.
    """

    TASKS_NECK = {
        1: "0cmH2O_normal_pulse",
        2: "4cmH2O_normal_pulse",
        3: "4cmH2O_apnea1_pulse",
        4: "4cmH2O_apnea2_pulse",
        5: "8cmH2O_normal_pulse",
        6: "8cmH2O_apnea1_pulse",
        7: "8cmH2O_apnea2_pulse"
    }


    for subject in range (1,21):
        for task_id, task_name in TASKS_NECK.items():
            file_name = f"Subject{subject}_{task_name}.csv"
            file_path = os.path.join(input_folder,
                                        f"Subject{subject}", file_name)
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found")
                continue
            if file_path.endswith(".csv"):
                print(f"Processing {file_path}...")
                plot_oximeter_file(file_path)

if __name__ == "__main__":
    TARGET_FOLDER = "physionet.org/files/respiratory-oximetry-apnoea/1.0.0/Neck_Pulse_Oximeter_Data" 
    
    if os.path.exists(TARGET_FOLDER):
        process_all_files(TARGET_FOLDER)
    else:
        print(f"Folder '{TARGET_FOLDER}' not found. Please update the TARGET_FOLDER path.")