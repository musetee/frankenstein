import os
import csv

def extract_prefixes_from_directory(directory):
    prefixes = set()
    for filename in os.listdir(directory):
        if filename.endswith('.nrrd'):
            prefix = filename.split('_')[0]
            prefixes.add(prefix)
    return sorted(prefixes)

def save_prefixes_to_csv(prefixes, output_csv_path):
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for prefix in prefixes:
            writer.writerow([os.path.join(directory, prefix)])

if __name__ == "__main__":
    directory = r"F:\yang_Projects\ICTUNET_torch\datasets\train"
    output_csv_path = r"F:\yang_Projects\ICTUNET_torch\data_table\train_all.csv"

    prefixes = extract_prefixes_from_directory(directory)
    save_prefixes_to_csv(prefixes, output_csv_path)

    print(f"CSV file with prefixes saved to: {output_csv_path}")
