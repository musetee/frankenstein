import re
import numpy as np

def parse_metrics_from_log_pattern1(file_path):
    """
    Parse SSIM, PSNR, and MAE values from the log file and calculate statistics.

    Parameters:
    - file_path (str): Path to the log file.

    Returns:
    - metrics (dict): Dictionary containing mean, std, and extreme values with corresponding pID and totalstep.
    """
    ssim_values = []
    psnr_values = []
    mae_values = []

    # Stores the full records for tracking extreme values
    records = []

    pattern = re.compile(
    r"pID\s(\S+)\]\s\[totalstep\s(\d+)\].*ssim:\s([\d\.]+)\]\s\[psnr:\s([\d\.]+)\]\s\[mae:\s([\d\.]+)"
    )

    # Read the log file and extract values
    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                pID = match.group(1)
                totalstep = int(match.group(2))
                ssim = float(match.group(3))
                psnr = float(match.group(4))
                mae = float(match.group(5))

                ssim_values.append(ssim)
                psnr_values.append(psnr)
                mae_values.append(mae)

                records.append({
                    "pID": pID,
                    "totalstep": totalstep,
                    "ssim": ssim,
                    "psnr": psnr,
                    "mae": mae
                })

    # Identify the highest and lowest metrics
    best_ssim = max(records, key=lambda x: x["ssim"])
    worst_ssim = min(records, key=lambda x: x["ssim"])
    best_psnr = max(records, key=lambda x: x["psnr"])
    worst_psnr = min(records, key=lambda x: x["psnr"])
    best_mae = min(records, key=lambda x: x["mae"])  # Lower MAE is better
    worst_mae = max(records, key=lambda x: x["mae"])

    # Calculate mean and std
    metrics = {
        "ssim_mean": np.mean(ssim_values),
        "ssim_std": np.std(ssim_values),
        "psnr_mean": np.mean(psnr_values),
        "psnr_std": np.std(psnr_values),
        "mae_mean": np.mean(mae_values),
        "mae_std": np.std(mae_values),
        "best_ssim": best_ssim,
        "worst_ssim": worst_ssim,
        "best_psnr": best_psnr,
        "worst_psnr": worst_psnr,
        "best_mae": best_mae,
        "worst_mae": worst_mae,
    }

    return metrics

def parse_metrics_from_log_pattern2(file_path):
    """
    Parse SSIM, PSNR, and MAE values from the log file and calculate statistics.

    Parameters:
    - file_path (str): Path to the log file.

    Returns:
    - metrics (dict): Dictionary containing mean, std, and extreme values with corresponding pID and totalstep.
    """
    ssim_values = []
    psnr_values = []
    mae_values = []

    # Stores the full records for tracking extreme values
    records = []

    pattern = re.compile(
    r"mean metrics (\d+), SSIM: ([\d\.]+), MAE: ([\d\.]+), PSNR: ([\d\.]+)"
    )

    # Read the log file and extract values
    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                ssim = float(match.group(2))
                psnr = float(match.group(3))
                mae = float(match.group(4))

                ssim_values.append(ssim)
                psnr_values.append(psnr)
                mae_values.append(mae)

                records.append({
                    "ssim": ssim,
                    "psnr": psnr,
                    "mae": mae
                })

    # Identify the highest and lowest metrics
    best_ssim = max(records, key=lambda x: x["ssim"])
    worst_ssim = min(records, key=lambda x: x["ssim"])
    best_psnr = max(records, key=lambda x: x["psnr"])
    worst_psnr = min(records, key=lambda x: x["psnr"])
    best_mae = min(records, key=lambda x: x["mae"])  # Lower MAE is better
    worst_mae = max(records, key=lambda x: x["mae"])

    # Calculate mean and std
    metrics = {
        "ssim_mean": np.mean(ssim_values),
        "ssim_std": np.std(ssim_values),
        "psnr_mean": np.mean(psnr_values),
        "psnr_std": np.std(psnr_values),
        "mae_mean": np.mean(mae_values),
        "mae_std": np.std(mae_values),
        "best_ssim": best_ssim,
        "worst_ssim": worst_ssim,
        "best_psnr": best_psnr,
        "worst_psnr": worst_psnr,
        "best_mae": best_mae,
        "worst_mae": worst_mae,
    }

    return metrics



val_log_folder = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\val_logs'
#val_log_folder = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\val_logs\ct2mr'
import os
# List all files and folders in the directory
all_files = os.listdir(val_log_folder)
# Filter to include only files
files = [os.path.join(val_log_folder, f) for f in all_files if os.path.isfile(os.path.join(val_log_folder, f))]
# Regular expression to extract metrics and identifiers

# Example usage
file_path = r"D:\Project\seg2med_Project\synthrad_results\results_all_eval\ct_compare_4_models_256\20241119_0052_Infer_ddpm2d_seg2med_val_log_512.txt"  # Replace with your log file path
file_path = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase1_mr\mr_synthrad_val_40_256.txt'
file_path = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\zhilin_results_testset10\resultsnifti\metrics_log.txt'
file_path = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase2_xcat_synthetic\20241119_0028_Infer_ddpm2d_seg2med_XCAT_CT_56Models_64slices_512\saved_logs\20241119_0028_Infer_ddpm2d_seg2med_val_log.txt'
file_path = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase1_ct_anika_4_models\Infer_ddpm2d_seg2med_anika_512_all\saved_logs\20241125_2234_Infer_ddpm2d_seg2med_val_log.txt'
file_path = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\val_logs\pix2pix_ct_anish_synthrad_512.txt'
files = [file_path]

for file_path in files:
    metrics = parse_metrics_from_log_pattern1(file_path)

    print(f"Evaluation Metrics of {file_path}:")
    # Print the results
    print("Evaluation Metrics:")
    print(f"SSIM Mean: {metrics['ssim_mean']:.4f}, Std: {metrics['ssim_std']:.4f}")
    print(f"PSNR Mean: {metrics['psnr_mean']:.4f}, Std: {metrics['psnr_std']:.4f}")
    print(f"MAE Mean: {metrics['mae_mean']:.4f}, Std: {metrics['mae_std']:.4f}\n")

    print("Highest and Lowest Metrics:")
    print(f"Best SSIM: {metrics['best_ssim']}")
    print(f"Worst SSIM: {metrics['worst_ssim']}")
    print(f"Best PSNR: {metrics['best_psnr']}")
    print(f"Worst PSNR: {metrics['worst_psnr']}")
    print(f"Best MAE: {metrics['best_mae']}")
    print(f"Worst MAE: {metrics['worst_mae']}")
