import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
def draw_overlap_image(folder, patient_id, organ_name, output_dir, slice_number):
    """
    Draw an overlap image for a slice of a certain organ.

    Args:
        folder (str): Path to the folder containing patient data.
        patient_id (str): Patient ID (e.g., "34438427_5").
        organ_name (str): Organ name (e.g., "liver").
        slice_number (int): Slice number to visualize.

    Returns:
        None
    """
    synthesized_slice, target_slice, slice_number = load_one_organ(folder, patient_id, organ_name, slice_number)
    # Normalize slices to binary masks (0 or 1)
    synthesized_mask = (synthesized_slice > 0).astype(np.uint8)
    target_mask = (target_slice > 0).astype(np.uint8)
    
    # Create an RGBA image
    height, width = synthesized_mask.shape
    rgba_image = np.zeros((height, width, 4), dtype=np.float32)  # RGBA

    # Set red for target mask and blue for synthesized mask
    rgba_image[..., 0] = target_mask  # Red channel
    rgba_image[..., 2] = synthesized_mask  # Blue channel
    rgba_image[..., 3] = np.clip(target_mask + synthesized_mask, 0, 1) * 0.5  # Alpha channel for translucency

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set the output file path
    output_file = os.path.join(output_dir, f"{patient_id}_{organ_name}_slice{slice_number}.png")
    
    # Plot and save the image with a black background
    plt.figure(figsize=(10, 10))
    plt.imshow(rgba_image, interpolation='none')
    plt.axis('off')  # Remove axes
    plt.gca().set_facecolor('black')  # Ensure background is black
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close()

    print(f"Saved translucent overlap image to: {output_file}")

def load_one_organ(folder, patient_id, organ_name, slice_number=None):
    synthesized_path = f"{folder}/{patient_id}_synthesized_volume_seg/{organ_name}.nii.gz"
    target_path = f"{folder}/{patient_id}_target_volume_seg/{organ_name}.nii.gz"
    
    # Load segmentation data
    synthesized_nii = nib.load(synthesized_path)
    target_nii = nib.load(target_path)
    synthesized_data = synthesized_nii.get_fdata()
    target_data = target_nii.get_fdata()

    if slice_number==None:
        # Find the slice with the largest segmentation area
        synthesized_areas = [np.sum(synthesized_data[:, :, i] > 0) for i in range(synthesized_data.shape[2])]
        target_areas = [np.sum(target_data[:, :, i] > 0) for i in range(target_data.shape[2])]
        largest_slice = np.argmax(np.maximum(synthesized_areas, target_areas))
        slice_number = largest_slice
    # Select the slice
    synthesized_slice = synthesized_data[:, :, slice_number]
    target_slice = target_data[:, :, slice_number]

    synthesized_slice = np.rot90(synthesized_slice, k=-1)  # Rotate clockwise
    synthesized_slice = np.fliplr(synthesized_slice)

    target_slice = np.rot90(target_slice, k=-1)  # Rotate clockwise
    target_slice = np.fliplr(target_slice)
    return synthesized_slice, target_slice, slice_number

def draw_overlap_2_organs_image(folder, patient_id, organ_name1, organ_name2, output_dir, slice_number):
    """
    Draw an overlap image for a slice of a certain organ.

    Args:
        folder (str): Path to the folder containing patient data.
        patient_id (str): Patient ID (e.g., "34438427_5").
        organ_name (str): Organ name (e.g., "liver").
        slice_number (int): Slice number to visualize.

    Returns:
        None
    """
    # Paths for synthesized and target organ segmentation
    synthesized_slice1, target_slice1, slice_number =load_one_organ(folder, patient_id, organ_name1, slice_number)
    synthesized_slice2, target_slice2, slice_number =load_one_organ(folder, patient_id, organ_name2, slice_number)

    synthesized_slice=synthesized_slice1+synthesized_slice2
    target_slice=target_slice1+target_slice2
    # Normalize slices to binary masks (0 or 1)
    synthesized_mask = (synthesized_slice > 0).astype(np.uint8)
    target_mask = (target_slice > 0).astype(np.uint8)
    
    # Create an RGBA image
    height, width = synthesized_mask.shape
    rgba_image = np.zeros((height, width, 4), dtype=np.float32)  # RGBA

    # Set red for target mask and blue for synthesized mask
    rgba_image[..., 0] = target_mask  # Red channel
    rgba_image[..., 2] = synthesized_mask  # Blue channel
    rgba_image[..., 3] = np.clip(target_mask + synthesized_mask, 0, 1) * 0.5  # Alpha channel for translucency

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set the output file path
    output_file = os.path.join(output_dir, f"{patient_id}_{organ_name1}_slice{slice_number}.png")
    
    # Plot and save the image with a black background
    plt.figure(figsize=(10, 10))
    plt.imshow(rgba_image, interpolation='none')
    plt.axis('off')  # Remove axes
    plt.gca().set_facecolor('black')  # Ensure background is black
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close()

    print(f"Saved translucent overlap image to: {output_file}")

import os

def dice_coefficient(seg1, seg2):
    """
    Calculate the Dice coefficient between two binary segmentations.

    Args:
        seg1 (np.ndarray): First segmentation (binary mask).
        seg2 (np.ndarray): Second segmentation (binary mask).

    Returns:
        float: Dice coefficient.
    """
    intersection = np.sum(seg1 * seg2)
    volume_sum = np.sum(seg1) + np.sum(seg2)
    return 2.0 * intersection / volume_sum if volume_sum != 0 else 0.0

from tqdm import tqdm
def calculate_dice(folder, valid_organs):
    """
    Calculate Dice coefficients for all valid organs across all patients.

    Args:
        folder (str): Path to the folder containing patient data.
        valid_organs (list): List of valid organ names to include in calculations.

    Returns:
        list: List of tuples (patient_id, organ_name, dice_coeff).
    """
    results = []

    # Iterate through each patient folder
    for patient_id in tqdm(os.listdir(folder)):
        if "_synthesized_volume_seg" in patient_id:
            # Get the corresponding target folder
            target_id = patient_id.replace("_synthesized_volume_seg", "_target_volume_seg")
            synthesized_dir = os.path.join(folder, patient_id)
            target_dir = os.path.join(folder, target_id)
            
            for organ_name in os.listdir(synthesized_dir):
                if organ_name.endswith(".nii.gz"):
                    organ_name = organ_name.replace(".nii.gz", "")
                    if organ_name in valid_organs:
                        # Load synthesized and target segmentation
                        synth_path = os.path.join(synthesized_dir, f"{organ_name}.nii.gz")
                        target_path = os.path.join(target_dir, f"{organ_name}.nii.gz")
                        synth_seg = nib.load(synth_path).get_fdata() > 0
                        target_seg = nib.load(target_path).get_fdata() > 0
                        
                        # Calculate Dice coefficient
                        dice = dice_coefficient(synth_seg, target_seg)
                        results.append((patient_id, organ_name, dice))
    
    return results

def calculate_dice_all_organs(folder):
    """
    Calculate Dice coefficients for all valid organs across all patients.

    Args:
        folder (str): Path to the folder containing patient data.

    Returns:
        list: List of tuples (patient_id, organ_name, dice_coeff).
    """
    results = []

    # Iterate through each patient folder
    for patient_id in tqdm(os.listdir(folder)):
        if "_synthesized_volume_seg" in patient_id:
            # Get the corresponding target folder
            target_id = patient_id.replace("_synthesized_volume_seg", "_target_volume_seg")
            synthesized_dir = os.path.join(folder, patient_id)
            target_dir = os.path.join(folder, target_id)
            
            for organ_name in os.listdir(synthesized_dir):
                if organ_name.endswith(".nii.gz"):
                    organ_name = organ_name.replace(".nii.gz", "")
                    # Load synthesized and target segmentation
                    synth_path = os.path.join(synthesized_dir, f"{organ_name}.nii.gz")
                    target_path = os.path.join(target_dir, f"{organ_name}.nii.gz")
                    synth_seg = nib.load(synth_path).get_fdata() > 0
                    target_seg = nib.load(target_path).get_fdata() > 0
                    
                    # Calculate Dice coefficient
                    dice = dice_coefficient(synth_seg, target_seg)

                    results.append((patient_id, organ_name, dice))
                    if dice > 0:
                        results.append((patient_id, organ_name, dice))
    # Save the results
    with open(os.path.join(folder, "dice_results.txt"), "w") as f:
        for patient_id, organ_name, dice in results:
            f.write(f"{patient_id}, {organ_name}, {dice:.4f}\n")
    return results

import pandas as pd

def calculate_mean_std_dice_per_organ(dice_results):
    """
    Calculate the mean Dice coefficient for each organ.

    Args:
        dice_results (list): List of tuples (patient_id, organ_name, dice_coeff).

    Returns:
        dict: A dictionary with organ names as keys and their mean Dice coefficients as values.
    """
    # Convert results to a pandas DataFrame for easy aggregation
    df = pd.DataFrame(dice_results, columns=["Patient_ID", "Organ", "Dice_Coeff"])

    # Group by organ and calculate the mean Dice coefficient
    mean_dice_per_organ = df.groupby("Organ")["Dice_Coeff"].mean().to_dict()
    std_dice_per_organ = df.groupby("Organ")["Dice_Coeff"].std().to_dict()
    return mean_dice_per_organ, std_dice_per_organ

import seaborn as sns
import pandas as pd

def draw_dice_map(dice_results, output_dir):
    """
    Draw a heatmap of Dice coefficients.

    Args:
        dice_results (list): List of tuples (patient_id, organ_name, dice_coeff).

    Returns:
        None
    """
    # Convert results to a DataFrame
    df = pd.DataFrame(dice_results, columns=["Patient_ID", "Organ", "Dice_Coeff"])

    # Pivot the DataFrame for a heatmap
    heatmap_data = df.pivot(index="Organ", columns="Patient_ID", values="Dice_Coeff")
    heatmap_data = heatmap_data.fillna(0)  # Fill missing values with 0
    # Exclude patients with any Dice coefficient equal to 0
    #valid_patients = heatmap_data.columns[(heatmap_data > 0).all(axis=0)]
    #heatmap_data = heatmap_data[valid_patients]

    patient_id_mapping = {pid: idx + 1 for idx, pid in enumerate(heatmap_data.columns)}
    heatmap_data = heatmap_data.rename(columns=patient_id_mapping)
    # Create output file path

    output_file = os.path.join(output_dir, f"dice_map.png")

    # Plot heatmap
    plt.figure(figsize=(40, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f", cbar_kws={'label': 'Dice Coefficient'})
    plt.xlabel("Patient ID")
    plt.ylabel("Organ")
    plt.title("Dice Coefficient Map")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def draw_dice_map_from_txt(results_file, output_dir):
    """
    Draw a heatmap of Dice coefficients from a saved results text file, excluding patients
    with any Dice coefficient equal to 0.

    Args:
        results_file (str): Path to the text file containing Dice results.
        output_dir (str): Directory to save the heatmap image.

    Returns:
        None
    """
    # Read the results from the text file into a DataFrame
    data = []

    with open(results_file, "r") as f:
        for line in f:
            patient_id, organ_name, dice = line.strip().split(", ")
            data.append((patient_id, organ_name, float(dice)))

    # Convert results to a DataFrame
    df = pd.DataFrame(data, columns=["Patient_ID", "Organ", "Dice_Coeff"])

    # Pivot the DataFrame for a heatmap
    heatmap_data = df.pivot(index="Organ", columns="Patient_ID", values="Dice_Coeff")
    heatmap_data = heatmap_data.fillna(0)  # Fill missing values with 0

    # Exclude patients with any Dice coefficient equal to 0
    valid_patients = heatmap_data.columns[(heatmap_data > 0).all(axis=0)]
    heatmap_data = heatmap_data[valid_patients]

    patient_id_mapping = {pid: idx + 1 for idx, pid in enumerate(heatmap_data.columns)}
    heatmap_data = heatmap_data.rename(columns=patient_id_mapping)
    # Create output file path
    output_file = os.path.join(output_dir, "dice_map.png")

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f", cbar_kws={'label': 'Dice Coefficient'})
    plt.xlabel("Patient ID")
    plt.ylabel("Organ")
    plt.title("Dice Coefficient Map")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()

    print(f"Dice coefficient heatmap saved to: {output_file}")

def extract_dice_results(file_path):
    """
    Extract Dice coefficient results from a text file and filter out entries with Dice = 0.

    Args:
        file_path (str): Path to the text file containing Dice coefficient results.

    Returns:
        list: List of tuples (patient_id, organ_name, dice_coeff).
    """
    results = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # Split the line into patient_id, organ_name, and dice
            parts = line.strip().split(", ")
            if len(parts) == 3:
                patient_id, organ_name, dice_str = parts
                dice_coeff = float(dice_str)
                
                # Exclude entries with Dice coefficient of 0
                if dice_coeff > 0:
                    results.append((patient_id, organ_name, dice_coeff))
    
    return results

import os
from tqdm import tqdm
# Path to the directory containing the folders
#directory = '/path/to/your/folder'

def rename_folders(directory):
    # Iterate through each folder in the directory
    for folder_name in tqdm(os.listdir(directory)):
        # Construct the full folder path
        old_path = os.path.join(directory, folder_name)
        
        # Check if it's a directory
        if os.path.isdir(old_path):
            # Remove '.nii.gz' from the folder name
            new_folder_name = folder_name.replace('.nii.gz', '')
            new_path = os.path.join(directory, new_folder_name)
            
            # Rename the folder
            os.rename(old_path, new_path)
            print(f"Renamed: {folder_name} -> {new_folder_name}")

steps = [
    #"step1_draw_overlap",
    #"step2_calculate_dice",
    #"step3_dice_map",
    #"step_optional_rename_folders",
    #"step4_get_all_dice_mean",
    "step4_get_all_dice_mean_from_txt",
]

anika_folder = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg\ddpm_anika_8_512_for_2_seg'
anish_folder = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg\ddpm_anish_512_for_2_seg'

output_dir = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg'

if "step1_draw_overlap" in steps:
    folder = anika_folder
    # Example usage
    draw_overlap_image(folder, "34438427_5", "stomach", output_dir, None)
    draw_overlap_image(folder, "34438427_5", "liver", output_dir, 50)
    draw_overlap_2_organs_image(folder, "34438427_5", "kidney_left", "kidney_right", output_dir, 13)
    draw_overlap_2_organs_image(folder, "34438427_5", "autochthon_left", "autochthon_right", output_dir, 13)
    draw_overlap_image(folder, "34438427_5", "colon", output_dir, 18)
    draw_overlap_image(folder, "34438427_5", "spleen", output_dir, 45)

    draw_overlap_image(folder, "34952781_4", "stomach", output_dir, None)
    draw_overlap_image(folder, "34952781_4", "liver", output_dir, 39)
    draw_overlap_2_organs_image(folder, "34952781_4", "kidney_left", "kidney_right", output_dir, 5)
    draw_overlap_2_organs_image(folder, "34952781_4", "autochthon_left", "autochthon_right", output_dir, 13)
    draw_overlap_image(folder, "34952781_4", "colon", output_dir, 8)
    draw_overlap_image(folder, "34952781_4", "spleen", output_dir, 69)

    draw_overlap_image(folder, "40094015_4", "stomach", output_dir, None)
    draw_overlap_image(folder, "40094015_4", "liver", output_dir, 74)
    draw_overlap_2_organs_image(folder, "40094015_4", "kidney_left", "kidney_right", output_dir, 44)
    draw_overlap_2_organs_image(folder, "40094015_4", "autochthon_left", "autochthon_right", output_dir, 26)
    draw_overlap_image(folder, "40094015_4", "colon", output_dir, 16)
    draw_overlap_image(folder, "40094015_4", "spleen", output_dir, 68)

    folder = anish_folder
    draw_overlap_image(folder, "382", "stomach", output_dir, None)
    draw_overlap_image(folder, "382", "liver", output_dir, None)
    draw_overlap_2_organs_image(folder, "382", "kidney_left", "kidney_right", output_dir, None)
    draw_overlap_2_organs_image(folder, "382", "autochthon_left", "autochthon_right", output_dir, None)
    draw_overlap_image(folder, "382", "colon", output_dir, None)
    draw_overlap_image(folder, "382", "spleen", output_dir, None)

    draw_overlap_image(folder, "380", "liver", output_dir, 210)
    draw_overlap_2_organs_image(folder, "380", "kidney_left", "kidney_right", output_dir, 153)
    draw_overlap_2_organs_image(folder, "380", "autochthon_left", "autochthon_right", output_dir, 13)
    draw_overlap_image(folder, "380", "colon", output_dir, 113)
    draw_overlap_image(folder, "380", "spleen", output_dir, 191)

    draw_overlap_image(folder, "3V23280", "stomach", output_dir, None)
    draw_overlap_image(folder, "V232", "stomach", output_dir, None)
    draw_overlap_image(folder, "V232", "liver", output_dir, None)
    draw_overlap_2_organs_image(folder, "V232", "kidney_left", "kidney_right", output_dir, None)
    draw_overlap_2_organs_image(folder, "V232", "autochthon_left", "autochthon_right", output_dir, None)
    draw_overlap_image(folder, "V232", "colon", output_dir, None)
    draw_overlap_image(folder, "V232", "spleen", output_dir, None)

    draw_overlap_image(folder, "381", "stomach", output_dir, None)
    draw_overlap_image(folder, "381", "liver", output_dir, None)
    draw_overlap_2_organs_image(folder, "381", "kidney_left", "kidney_right", output_dir, None)
    draw_overlap_2_organs_image(folder, "381", "autochthon_left", "autochthon_right", output_dir, None)
    draw_overlap_image(folder, "381", "colon", output_dir, None)
    draw_overlap_image(folder, "381", "spleen", output_dir, None)

if "step_optional_rename_folders" in steps:
    new_folder = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg\ddpm_anika_more_512_for_2_seg'
    rename_folders(new_folder)

if "step2_calculate_dice" in steps:
    valid_organs = ["liver", "colon", "spleen", "kidney_left", "kidney_right", 
                    "stomach", "autochthon_left", "autochthon_right", "colon", "aorta", 
                    "costal_cartilages", "duodenum", "inferior_vena_cava", "lung_lower_lobe_left", "lung_lower_lobe_right",
                    "lung_middle_lobe_right", "pancreas"]  # Specify valid organs

    #dice_results_anish = calculate_dice(anika_folder, valid_organs)
    #dice_results_anika = calculate_dice(anish_folder, valid_organs)
    #dice_results = dice_results_anish + dice_results_anika

    anika_10_more_folder = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg\ddpm_anika_10_4_512_for_2_seg'
    dice_results = calculate_dice(anika_10_more_folder, valid_organs)

    # Save the results
    with open(os.path.join(output_dir, "dice_results.txt"), "w") as f:
        for patient_id, organ_name, dice in dice_results:
            f.write(f"{patient_id}, {organ_name}, {dice:.4f}\n")

if "step3_dice_map" in steps:
    anika_file_8 = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg\dice_results_8.txt'
    anika_file_10_1 = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg\dice_results_10_1.txt'
    anika_file_10_2 = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg\dice_results_10_2.txt'
    anika_file_10_3 = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg\dice_results_10_3.txt'
    anika_file_10_4 = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg\dice_results_10_4.txt'
    anish_file = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg\dice_results_anish_10.txt'    

    # Assume dice_results is the output from calculate_dice_all_organs
    dice_results_anika_8 = extract_dice_results(anika_file_8)
    dice_results_anika_10_1 = extract_dice_results(anika_file_10_1)
    dice_results_anika_10_2 = extract_dice_results(anika_file_10_2)
    dice_results_anika_10_3 = extract_dice_results(anika_file_10_3)
    dice_results_anika_10_4 = extract_dice_results(anika_file_10_4)
    dice_results_anish = extract_dice_results(anish_file)
    dice_results = dice_results_anika_8 + dice_results_anika_10_1 + dice_results_anika_10_2 + dice_results_anika_10_3 + dice_results_anika_10_4 + dice_results_anish
    # Example usage
    draw_dice_map(dice_results, output_dir)
    # draw_dice_map_from_txt(os.path.join(output_dir, "dice_results_all.txt"), output_dir)

if "step4_get_all_dice_mean" in steps:
    anika_folder_8 = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg\ddpm_anika_8_512_for_2_seg'
    anika_folder_10_1 = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg\ddpm_anika_10_1_512_for_2_seg'
    anika_folder_10_2 = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg\ddpm_anika_10_2_512_for_2_seg'
    anika_folder_10_3 = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg\ddpm_anika_10_3_512_for_2_seg'
    anika_folder_10_4 = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg\ddpm_anika_10_4_512_for_2_seg'
    anish_folder = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg\ddpm_anish_512_for_2_seg'    

    # Assume dice_results is the output from calculate_dice_all_organs
    dice_results_anika_8 = calculate_dice_all_organs(anika_folder_8)
    dice_results_anika_10_1 = calculate_dice_all_organs(anika_folder_10_1)
    dice_results_anika_10_2 = calculate_dice_all_organs(anika_folder_10_2)
    dice_results_anika_10_3 = calculate_dice_all_organs(anika_folder_10_3)
    dice_results_anika_10_4 = calculate_dice_all_organs(anika_folder_10_4)
    dice_results_anish = calculate_dice_all_organs(anish_folder)

    dice_results = dice_results_anika_8 + dice_results_anika_10_1 + dice_results_anika_10_2 + dice_results_anika_10_3 + dice_results_anika_10_4 + dice_results_anish
    
    
    # Calculate mean Dice coefficients per organ
    mean_dice, std_dice = calculate_mean_std_dice_per_organ(dice_results)

    # Print the results
    for organ in mean_dice.keys():
        print(f"{organ}\t{mean_dice[organ]:.4f}\t{std_dice[organ]:.4f}\n")

    # Save the mean Dice coefficients to a text file
    output_file = os.path.join(output_dir, "mean_dice_per_organ.txt")
    with open(output_file, "w") as f:
        for organ in mean_dice.keys():
            f.write(f"{organ}\t{mean_dice[organ]:.4f}\t{std_dice[organ]:.4f}\n")

    print(f"Mean Dice coefficients saved to: {output_file}")

if "step4_get_all_dice_mean_from_txt" in steps:
    anika_file_8 = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg\dice_results_8.txt'
    anika_file_10_1 = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg\dice_results_10_1.txt'
    anika_file_10_2 = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg\dice_results_10_2.txt'
    anika_file_10_3 = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg\dice_results_10_3.txt'
    anika_file_10_4 = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg\dice_results_10_4.txt'
    anish_file = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_secondseg\dice_results_anish_10.txt'    

    # Assume dice_results is the output from calculate_dice_all_organs
    dice_results_anika_8 = extract_dice_results(anika_file_8)
    dice_results_anika_10_1 = extract_dice_results(anika_file_10_1)
    dice_results_anika_10_2 = extract_dice_results(anika_file_10_2)
    dice_results_anika_10_3 = extract_dice_results(anika_file_10_3)
    dice_results_anika_10_4 = extract_dice_results(anika_file_10_4)
    dice_results_anish = extract_dice_results(anish_file)
    dice_results = dice_results_anika_8 + dice_results_anika_10_1 + dice_results_anika_10_2 + dice_results_anika_10_3 + dice_results_anika_10_4 + dice_results_anish
    
    # Calculate mean Dice coefficients per organ
    mean_dice, std_dice = calculate_mean_std_dice_per_organ(dice_results)

    # Print the results
    for organ in mean_dice.keys():
        print(f"{organ}\t{mean_dice[organ]:.4f}\t{std_dice[organ]:.4f}\n")

    # Save the mean Dice coefficients to a text file
    output_file = os.path.join(output_dir, "mean_dice_per_organ.txt")
    with open(output_file, "w") as f:
        for organ in mean_dice.keys():
            f.write(f"{organ}\t{mean_dice[organ]:.4f}\t{std_dice[organ]:.4f}\n")

    print(f"Mean Dice coefficients saved to: {output_file}")