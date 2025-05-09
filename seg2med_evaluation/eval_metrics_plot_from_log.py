import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
# Function to parse training result files
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Function to parse training result files
def parse_results(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'\[pID (.*?)\].*?ssim: ([\d.]+).*?psnr: ([\d.]+).*?mae: ([\d.]+)', line)
            if match:
                patient_id, ssim, psnr, mae = match.groups()
                data.append({
                    "patient_id": patient_id,
                    "ssim": float(ssim),
                    "psnr": float(psnr),
                    "mae": float(mae)
                })
    return pd.DataFrame(data)

# Function to calculate mean metrics per patient
def calculate_means(data):
    return data.groupby("patient_id", as_index=False).mean()

# Function to calculate means and stds for a DataFrame
def calculate_means_all(dataframe):
    metrics = ['ssim', 'psnr', 'mae']
    summary = {}
    for metric in metrics:
        mean = dataframe[metric].mean()
        std = dataframe[metric].std()
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_std"] = std
        print(f"{metric}, mean {mean}, std {std}")
    return summary

# Function to anonymize patient IDs
def anonymize_patient_ids(results):
    anonymized_results = {}
    patient_map = {}
    current_index = 1

    for model, datasets in results.items():
        anonymized_results[model] = {}
        for dataset, data in datasets.items():
            anonymized_data = data.copy()
            anonymized_data["patient_id"] = anonymized_data["patient_id"].map(
                lambda pid: patient_map.setdefault(pid, current_index + len(patient_map))
            )
            anonymized_results[model][dataset] = anonymized_data

    return anonymized_results

# Function to anonymize dataset names
def anonymize_dataset_names(datasets):
    # return {dataset: f"Dataset {i+1}" for i, dataset in enumerate(datasets)}
    return {dataset: f"Dataset {i+1}" for i, dataset in enumerate(datasets)}

# Load results for each model and dataset
def load_and_process_data(base_path, models, datasets):
    results = {}
    parsed_all_datasets = pd.DataFrame()
    for model in models:
        print(f"evaluate for {model}")
        results[model] = {}
        for dataset in datasets:
            file_path = os.path.join(base_path, f"{model}_{dataset}.txt")
            if os.path.exists(file_path):
                parsed_data = parse_results(file_path)
                parsed_all_datasets = pd.concat([parsed_all_datasets, parsed_data], ignore_index=True)
                results[model][dataset] = calculate_means(parsed_data)
        _ = calculate_means_all(parsed_all_datasets)
    return results

# Load results for each model and dataset
def load_and_process_data_slices(base_path, models, datasets):
    results = {}
    parsed_all_datasets = pd.DataFrame()
    for model in models:
        print(f"evaluate for {model}")
        results[model] = {}
        for dataset in datasets:
            file_path = os.path.join(base_path, f"{model}_{dataset}.txt")
            if os.path.exists(file_path):
                parsed_data = parse_results(file_path)
                parsed_all_datasets = pd.concat([parsed_all_datasets, parsed_data], ignore_index=True)
                results[model][dataset] = parsed_data
        _ = calculate_means_all(parsed_all_datasets)
    return results

# Plot and save metrics diagrams with simplified legend and background colors
def plot_and_save_metrics_with_background(results, metrics, models, datasets, dataset_map, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory if it doesn't exist

    # Define a consistent color map for the models
    color_map = {
        "unet_ct": "blue",
        "cyclegan_ct": "green",
        "pix2pix_ct": "red",
        "ddpm_ct": "purple"
    }

    # Define background colors for datasets
    background_colors = ["lightblue", "lightgreen", "lightpink"]

    for metric in metrics:
        plt.figure(figsize=(12, 8))

        # Add background colors for datasets
        for i, dataset in enumerate(datasets):
            plt.axvspan(i * 100, (i + 1) * 100, color=background_colors[i], alpha=0.2, label=dataset_map[dataset])

        # Plot data for each model and dataset
        for model in models:
            for i, dataset in enumerate(datasets):
                if dataset in results[model]:
                    data = results[model][dataset]
                    anonymized_dataset = dataset_map[dataset]
                    plt.plot(
                        data["patient_id"] + i * 100,  # Offset x-axis by dataset
                        data[metric],
                        marker='o',
                        color=color_map[model],
                        label=model if i == 0 else None  # Add model to legend only once
                    )

        plt.title(f"{metric.upper()} Comparison Across Models and Datasets")
        plt.xlabel("Anonymized Patient Index")
        plt.ylabel(metric.upper())
        plt.legend(loc='upper left', fontsize='small')  # Simplified legend with unique models
        plt.grid(True)

        # Save the plot as a PNG file
        file_name = f"{metric}_comparison_background.png"
        plt.savefig(os.path.join(output_dir, file_name), dpi=300)
        plt.close()  # Close the figure to free memory
# Plot and save metrics diagrams with continuous indices

def plot_and_save_metrics_without_gaps(results, metrics, models, datasets, dataset_map, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory if it doesn't exist

    # Define a consistent color map for the models
    color_map = {
        "unet_ct": "darkcyan",
        "cyclegan_ct": "gold",
        "pix2pix_ct": "slateblue",
        "ddpm_ct": "coral"
    }

    marker_map = {
        "unet_ct": "o",
        "cyclegan_ct": "v",
        "pix2pix_ct": "^",
        "ddpm_ct": "s"
    }

    # Define background colors for datasets
    background_colors = ["lightblue", "lightgreen", "lightpink"]

    label_loc_map = {
        "mae": "upper right",
        "psnr": "lower right",
        "ssim": "lower right",
    }
    for metric in metrics:
        plt.figure(figsize=(12, 8))

        # Add background colors for datasets
        cumulative_index = 0
        for i, dataset in enumerate(datasets):
            dataset_length = len(results[models[0]][dataset])
            print(dataset_length)
            anonymized_dataset = dataset_map[dataset]
            plt.axvspan(
                cumulative_index, cumulative_index + dataset_length, 
                color=background_colors[i], alpha=0.2, label= anonymized_dataset #f"Dataset {i+1}" # if metric == metrics[0] else None
            )
            cumulative_index += dataset_length

        # Plot data for each model and dataset
        cumulative_index = 0
        for model in models:
            for i, dataset in enumerate(datasets):
                if dataset in results[model]:
                    data = results[model][dataset]
                    
                    plt.plot(data["patient_id"], data[metric], marker=marker_map[model],
                             color=color_map[model], label=model if i == 0 else None)

        plt.title(f"{metric.upper()} Comparison Across Models and Datasets")
        plt.xlabel("Patient Index")
        plt.ylabel(metric.upper())
        plt.legend(loc=label_loc_map[metric], fontsize='small')  # Simplified legend with unique models
        plt.grid(True)

        # Save the plot as a PNG file
        file_name = f"{metric}_comparison_continuous.png"
        plt.savefig(os.path.join(output_dir, file_name), dpi=300)
        plt.close()  # Close the figure to free memory

def plot_and_save_boxplots_with_points(results, metrics, models, datasets, dataset_map, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory if it doesn't exist

    # Define dataset-specific colors for scatter points
    dataset_colors = ["lightblue", "lightgreen", "lightpink"]
    label_loc_map = {
        "mae": "upper right",
        "psnr": "lower right",
        "ssim": "lower right",
    }
    for metric in metrics:
        plt.figure(figsize=(12, 8))

        # Prepare data for boxplot
        boxplot_data = []
        scatter_positions = []  # To offset points for each dataset
        all_points = []         # To collect points for scatter

        for model in models:
            model_data = []
            model_points = []
            for i, dataset in enumerate(datasets):
                if dataset in results[model]:
                    model_data.extend(results[model][dataset][metric])
                    # Offset points to avoid overlap: center +- small offsets
                    positions = [len(boxplot_data) + 1 + (i - 1) * 0.2 for _ in results[model][dataset][metric]]
                    model_points.append((positions, results[model][dataset][metric], dataset))
            boxplot_data.append(model_data)
            scatter_positions.append(model_points)

        # Create the boxplot
        box = plt.boxplot(boxplot_data, patch_artist=True, vert=True, labels=models)

        # Apply same semi-transparent color to all boxes
        box_color = "lightgray"
        for patch in box['boxes']:
            patch.set_facecolor(box_color)
            patch.set_alpha(0.5)

        # Plot individual data points for each dataset
        plotted_labels = set()  # To track labels already added to the legend
        for model_points in scatter_positions:
            for positions, values, dataset in model_points:
                label = dataset_map[dataset]
                if label not in plotted_labels:
                    plt.scatter(positions, values, color=dataset_colors[datasets.index(dataset)], 
                                alpha=0.8, s=30, label=label)
                    plotted_labels.add(label)
                else:
                    plt.scatter(positions, values, color=dataset_colors[datasets.index(dataset)], alpha=0.8, s=30)

        # Avoid duplicate labels in the legend
        plt.legend(loc=label_loc_map[metric], fontsize="small")

        plt.title(f"{metric.upper()} Boxplot Across Models")
        plt.xlabel("Model")
        plt.ylabel(metric.upper())
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save the boxplot as a PNG file
        file_name = f"{metric}_boxplot_with_points.png"
        plt.savefig(os.path.join(output_dir, file_name), dpi=300)
        plt.close()  # Close the figure to free memory

def plot_and_save_boxplots_with_points_beside(results, metrics, models, datasets, dataset_map, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory if it doesn't exist

    # Define dataset-specific colors for scatter points
    dataset_colors = ["lightblue", "lightgreen", "lightpink"]
    label_loc_map = {
        "mae": "upper right",
        "psnr": "lower right",
        "ssim": "lower right",
    }

    for metric in metrics:
        plt.figure(figsize=(12, 8))

        # Prepare data for boxplot
        boxplot_data = []
        scatter_positions = []  # To offset points for each dataset
        all_points = []         # To collect points for scatter

        for model_idx, model in enumerate(models):
            model_data = []
            model_points = []
            for dataset_idx, dataset in enumerate(datasets):
                if dataset in results[model]:
                    data = results[model][dataset][metric]
                    model_data.extend(data)
                    # Offset points horizontally to the side of the boxplot
                    positions = [model_idx + 1.35 + (dataset_idx - 1) * 0.1 for _ in data]
                    model_points.append((positions, data, dataset))
            boxplot_data.append(model_data)
            scatter_positions.append(model_points)

        # Create the boxplot
        box = plt.boxplot(boxplot_data, patch_artist=True, vert=True, labels=models)

        # Apply same semi-transparent color to all boxes
        box_color = "lightgray"
        for patch in box['boxes']:
            patch.set_facecolor(box_color)
            patch.set_alpha(0.5)

        # Plot individual data points for each dataset
        plotted_labels = set()  # To track labels already added to the legend
        for model_points in scatter_positions:
            for positions, values, dataset in model_points:
                label = dataset_map[dataset]
                if label not in plotted_labels:
                    plt.scatter(positions, values, color=dataset_colors[datasets.index(dataset)], 
                                alpha=0.8, s=30, label=label)
                    plotted_labels.add(label)
                else:
                    plt.scatter(positions, values, color=dataset_colors[datasets.index(dataset)], alpha=0.8, s=30)

        # Avoid duplicate labels in the legend
        plt.legend(loc=label_loc_map[metric], fontsize="small", frameon=False)

        plt.title(f"{metric.upper()} Boxplot Across Models")
        plt.xlabel("Model")
        plt.ylabel(metric.upper())
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the boxplot as a PNG file
        file_name = f"{metric}_boxplot_with_points_beside.png"
        plt.savefig(os.path.join(output_dir, file_name), dpi=300)
        plt.close()  # Close the figure to free memory


def plot_and_save_boxplots_with_datasets(results, metrics, models, datasets, dataset_map, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory if it doesn't exist

    # Define dataset-specific colors
    dataset_colors = ["lightblue", "lightgreen", "lightpink"]
    label_loc_map = {
        "mae": "upper right",
        "psnr": "lower right",
        "ssim": "lower right",
    }
    for metric in metrics:
        plt.figure(figsize=(12, 8))

        # Prepare data for boxplots
        all_boxplot_positions = []
        boxplot_data = {model: {"overall": [], "datasets": {}} for model in models}
        for dataset in datasets:
            for model in models:
                if dataset in results[model]:
                    data = results[model][dataset][metric]
                    boxplot_data[model]["overall"].extend(data)  # Aggregate for overall
                    boxplot_data[model]["datasets"][dataset] = data

        # Position counter
        current_position = 1
        position_map = {}

        # Draw boxplots
        for model in models:
            positions = []

            # Overall data
            overall_positions = [current_position]
            overall_box = plt.boxplot(
                [boxplot_data[model]["overall"]],
                patch_artist=True,
                positions=overall_positions,
                widths=0.5,
            )
            for patch in overall_box['boxes']:
                patch.set_facecolor("gray")
                patch.set_alpha(0.5)
            positions.extend(overall_positions)

            # Dataset-specific boxplots
            dataset_positions = []
            for i, dataset in enumerate(datasets):
                dataset_position = current_position + i + 1
                dataset_box = plt.boxplot(
                    [boxplot_data[model]["datasets"].get(dataset, [])],
                    patch_artist=True,
                    positions=[dataset_position],
                    widths=0.5,
                )
                for patch in dataset_box['boxes']:
                    patch.set_facecolor(dataset_colors[i])
                    patch.set_alpha(0.5)
                dataset_positions.append(dataset_position)
            positions.extend(dataset_positions)

            # Map positions for X-axis
            position_map[model] = positions
            current_position += len(datasets) + 2  # Add spacing for the next model

        # Add X-axis labels
        model_positions = [positions[len(datasets) // 2] for positions in position_map.values()]
        plt.xticks(model_positions, models, rotation=0)

        # Add a legend
        legend_labels = ["Overall Results"] + [dataset_map[dataset] for dataset in datasets]
        legend_patches = [plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.5)] + [
            plt.Rectangle((0, 0), 1, 1, color=dataset_colors[i], alpha=0.5) for i in range(len(datasets))
        ]
        plt.legend(legend_patches, legend_labels, loc=label_loc_map[metric], fontsize="small", frameon=False)

        # Plot settings
        plt.title(f"{metric.upper()} Boxplots Across Models and Datasets", fontsize=14)
        plt.xlabel("Model", fontsize=12)
        plt.ylabel(metric.upper(), fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the boxplot as a PNG file
        file_name = f"{metric}_boxplot_with_datasets_tight.png"
        plt.savefig(os.path.join(output_dir, file_name), dpi=300)
        plt.close()  # Close the figure to free memory



# Base path where your text files are located
base_path = "D:\\Project\\seg2med_Project\\synthrad_results\\results_all_eval\\val_logs"  # Replace with the correct path
models = ["unet_ct", "cyclegan_ct", "pix2pix_ct", "ddpm_ct"]  # Replace with your model names
# Define a consistent color map for the models
color_map = {
    "unet_ct": "blue",
    "cyclegan_ct": "green",
    "pix2pix_ct": "red",
    "ddpm_ct": "purple"
}
datasets = ["synthrad_256", "anish_256", "anika_256"]  # Replace with your dataset names
metrics = ["ssim", "psnr", "mae"]
output_dir = "D:\\Project\\seg2med_Project\\synthrad_results\\results_all_eval"
# Process and plot data
results = load_and_process_data(base_path, models, datasets)

anonymized_results = anonymize_patient_ids(results)

# Map datasets to anonymized names
dataset_map = anonymize_dataset_names(datasets)
dataset_map = {"synthrad_256": "SynthRAD Pelvis", "anish_256": "Internal Abdomen", "anika_256": "M2OLIE Abdomen"}
#print(anonymized_results)
#plot_and_save_metrics_without_gaps(anonymized_results, metrics, models, datasets,dataset_map, output_dir)
#plot_and_save_boxplots_with_points(results, metrics, models, datasets, dataset_map, output_dir)
plot_and_save_boxplots_with_points_beside(results, metrics, models, datasets, dataset_map, output_dir)
#plot_and_save_boxplots_with_datasets(results, metrics, models, datasets, dataset_map, output_dir)