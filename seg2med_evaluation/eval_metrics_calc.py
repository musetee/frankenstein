import nibabel as nib
import os
import sys
sys.path.append('./synthrad_conversion')
from utils.image_metrics import ImageMetrics
def calculate_mask_metrices(val_output, val_labels, val_masks,
                            log_file_overall, val_step, dynamic_range = [-1024., 3000.], printoutput=False):
    metricsCalc=ImageMetrics(dynamic_range)

    if val_masks is None:
        val_ssim = metricsCalc.ssim(val_output, val_labels) # 
        val_mae = metricsCalc.mae(val_output, val_labels)
        val_psnr = metricsCalc.psnr(val_output, val_labels)
    else:
        val_ssim = metricsCalc.ssim(val_output, val_labels, val_masks) # 
        val_mae = metricsCalc.mae(val_output, val_labels, val_masks)
        val_psnr = metricsCalc.psnr(val_output, val_labels, val_masks)

    val_metrices = {
        'ssim': val_ssim,
        'mae': val_mae,
        'psnr': val_psnr,
    }

    if printoutput:
        print(f"mean ssim {val_step}: {val_metrices['ssim']}") #:.4f
        print(f"mean mae {val_step}: {val_metrices['mae']}")
        print(f"mean psnr {val_step}: {val_metrices['psnr']}")

    #output_val_log('mean', val_step, val_log_file=log_file_overall, val_metrices=val_metrices)
    ssim = val_metrices.get('ssim', 0)
    mae = val_metrices.get('mae', 0)
    psnr = val_metrices.get('psnr', 0)
    with open(log_file_overall, 'a') as f:
        f.write(f'mean metrics {val_step}, SSIM: {ssim}, MAE: {mae}, PSNR: {psnr}\n')
    return val_metrices


def eval_metrics_calc(synth_nii_file_path, target_nii_file_path):
    # Load synthetic and target NIfTI data
    if synth_nii_file_path.suffix=='.nii' or synth_nii_file_path.suffix=='.nii.gz':
        synth_nii_data = nib.load(synth_nii_file_path)
        target_nii_data = nib.load(target_nii_file_path)
        # Extract volume data as NumPy arrays
        synth_volume = synth_nii_data.get_fdata()
        target_volume = target_nii_data.get_fdata()
    elif synth_nii_file_path.suffix=='.nrrd':
    # Load the NRRD files
        target_volume, _ = nrrd.read(target_nii_file_path)
        synth_volume, _ = nrrd.read(synth_nii_file_path)

    # Check if the two volumes have the same shape
    if synth_volume.shape != target_volume.shape:
        raise ValueError("Synthetic and target volumes must have the same shape.")

    # Number of slices in the volume
    slice_num = synth_volume.shape[-1]

    # Create a log file in the same directory as the synthetic NIfTI file
    metrics_log_file = os.path.join(os.path.dirname(os.path.dirname(synth_nii_file_path)), "metrics_log.txt")
    metrics_log_file_mean = os.path.join(os.path.dirname(os.path.dirname(synth_nii_file_path)), "metrics_log_mean.txt")
    # Dynamic range for metrics
    dynamic_range = [0., 255.]

    # Initialize accumulators for mean metrics
    total_eval_step = 0
    ssim_sum, psnr_sum, mae_sum = 0, 0, 0

    # Iterate through slices
    for slice_idx in range(slice_num):
        # Extract the corresponding slices
        synth_image = synth_volume[:, :, slice_idx]
        target_image = target_volume[:, :, slice_idx]

        # Calculate metrics for the current slice
        metrics = calculate_mask_metrices(
            synth_image,
            target_image,
            val_masks=None,
            log_file_overall=metrics_log_file,
            val_step=f"{total_eval_step}",
            dynamic_range=dynamic_range,
            printoutput=False
        )

        # Update total evaluation steps
        total_eval_step += 1

        # Accumulate metrics
        ssim_sum += metrics['ssim']
        psnr_sum += metrics['psnr']
        mae_sum += metrics['mae']

    # Calculate mean metrics
    mean_ssim = ssim_sum / slice_num
    mean_psnr = psnr_sum / slice_num
    mean_mae = mae_sum / slice_num

    # Log the mean metrics
    with open(metrics_log_file_mean, "a") as log_file:
        log_file.write("\nMean Metrics:\n")
        log_file.write(f"SSIM: {mean_ssim:.4f}\n")
        log_file.write(f"PSNR: {mean_psnr:.4f}\n")
        log_file.write(f"MAE: {mean_mae:.4f}\n")

    return {"mean_ssim": mean_ssim, "mean_psnr": mean_psnr, "mean_mae": mean_mae}

import os
import nrrd
from pathlib import Path

# Define the function to evaluate metrics for each pair of NRRD files
def eval_metrics_for_nrrd(root_dir):
    results = []

    # Traverse the directory structure
    for subdir in sorted(Path(root_dir).iterdir()):  # Iterate through numbered subdirectories
        if subdir.is_dir():
            target_file = None
            synth_file = None

            # Look for target and synthesized NRRD files in the subdirectory
            for file in sorted(subdir.iterdir()):
                if file.suffix == ".nrrd":
                    if "_synthesis" in file.name:
                        synth_file = file
                    elif "_mask" not in file.name:
                        target_file = file
            
            # If both target and synthesized files are found, process them
            if target_file and synth_file:
                print(f"Evaluating: Target={target_file.name}, Synthesized={synth_file.name}")

                # Call the eval_metrics_calc function to evaluate metrics
                metrics = eval_metrics_calc(synth_file, target_file)
                results.append({"target": str(target_file), "synthesized": str(synth_file), "metrics": metrics})
            else:
                print(f"Missing files in {subdir}. Skipping...")

    return results




if __name__ == '__main__':
    # Define the root directory of your data
    root_dir = r"D:\Project\seg2med_Project\synthrad_results\results_all_eval\zhilin_results_testset10\resultsnifti"

    # Run the evaluation
    evaluation_results = eval_metrics_for_nrrd(root_dir)

    # Print the results
    for result in evaluation_results:
        print(f"Target: {result['target']}, Synthesized: {result['synthesized']}, Metrics: {result['metrics']}")