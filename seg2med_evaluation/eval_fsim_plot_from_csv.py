import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
# Specify the folder containing your files
folder_path = r"D:\Project\seg2med_Project\SynthRad_GAN\seg2med_evaluation\MedicalImageEvaluation\Results1201_XCATCT\FSIM"
output_dir = r"D:\Project\seg2med_Project\synthrad_results\results_all_eval\output_png"
# List all files in the folder
file_list = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

# Initialize a dictionary to store mean FSIM for each file
mean_fsim = {}

# Process each file
for file in file_list:
    print("read: ", file)
    file_path = os.path.join(folder_path, file)
    # Read the FSIM values
    data = pd.read_csv(file_path, header=None, names=["FSIM"])
    # Calculate mean FSIM
    #print(data["FSIM"])
    meanvalue = np.mean(pd.to_numeric(data["FSIM"], errors="coerce"))
    #print(meanvalue)
    mean_fsim[file] = meanvalue

# Create a DataFrame from the results
results_df = pd.DataFrame(list(mean_fsim.items()), columns=["File", "Mean_FSIM"])
results_df["Patient"] = results_df["File"].str.extract(r"Model(\d+)_")[0].astype(str)
results_df = results_df.sort_values("Patient")

# Plot the mean FSIM values as a line chart
plt.figure(figsize=(10, 6))
plt.plot(results_df["Patient"], results_df["Mean_FSIM"], marker="o")
plt.title("Mean FSIM for Each XCAT Phantom")
plt.xlabel("XCAT Phantom ID")
plt.ylabel("Mean FSIM")
# Rotate x-axis labels vertically
plt.xticks(rotation=90)
plt.grid()
# Save the plot as a PNG file
file_name = f"fsim_xcat.png"
plt.savefig(os.path.join(output_dir, file_name), dpi=300)
plt.close()  # Close the figure to free memory
