from eval_metrics_plot_from_log import load_and_process_data, anonymize_patient_ids,load_and_process_data_slices
import numpy as np
import scipy.stats as stats
from scipy.stats import kurtosis, skew, mannwhitneyu

def perform_anova(results, datasets, metrics, models):
    p_values = {}
    data_per_model_all_datasets = []
    for metric in metrics:
        for dataset in datasets:
            # Prepare a list of metric values for each model
            data_per_model = [results[model][dataset][metric].dropna() for model in models]
            data_per_model_all_datasets =data_per_model_all_datasets + data_per_model
        # Perform one-way ANOVA test
        f_stat, p_value = stats.f_oneway(*data_per_model_all_datasets)
        p_values[metric] = p_value
        
    return p_values

def perform_anova_each_model(results, datasets, metrics, models):
    p_values = {}
    for model in models:
        p_values[model] = {}
        for metric in metrics:
            data_per_model_seen_datasets = []
            data_per_model_unseen_datasets = []
            for dataset in datasets:
                # Prepare a list of metric values for each model
                data_per_model = results[model][dataset][metric].values.flatten().tolist()
                if "synthrad" in dataset or "anish" in dataset: 
                    data_per_model_seen_datasets = data_per_model_seen_datasets + data_per_model
                if "anika" in dataset:
                    data_per_model_unseen_datasets = data_per_model_unseen_datasets + data_per_model
            
            skewness = skew(data_per_model_seen_datasets)
            kurt = kurtosis(data_per_model_unseen_datasets)  # Default is Fisher’s definition; add `fisher=False` for Pearson’s definition
            print(f"Skewness: {skewness}")
            print(f"Kurtosis: {kurt}")
            #data_per_model_seen_datasets = data_per_model_unseen_datasets[:20]
            #data_per_model_unseen_datasets = data_per_model_unseen_datasets[21:]
            # print(data_per_model_seen_datasets)
            # print(data_per_model_unseen_datasets)
            print("group 1 length: ", len(data_per_model_seen_datasets))
            print("group 2 length: ", len(data_per_model_unseen_datasets))
            # p_value = perform_welchs_ttest(data_per_model_seen_datasets, data_per_model_unseen_datasets)
            stat, p_value = mannwhitneyu(data_per_model_seen_datasets, data_per_model_unseen_datasets, alternative='two-sided')
            # print(f"U Statistic = {stat}, p-value = {p_value}")
            # Perform one-way ANOVA test
            p_values[model][metric] = p_value

            data_per_model_all = data_per_model_unseen_datasets + data_per_model_seen_datasets
            print(f"{model} {metric}, mean {np.mean(data_per_model_all)}, std {np.std(data_per_model_all)}")
    return p_values

def get_data_for_model(results, model, datasets):
    data_per_model = []
    for dataset in datasets:
        # Prepare a list of metric values for each model
        data_per_model = results[model][dataset][metric].values.flatten().tolist()
        data_per_model = data_per_model + data_per_model
    return data_per_model
def perform_anova_two_models(results, datasets, metrics, models=["unet_ct", "ddpm_ct"]):
    model1=models[0]
    model2=models[1]
    p_values = {}
    p_values[model] = {}
    
    for metric in metrics:
        data_model1 = get_data_for_model(results, model1, datasets)
        data_model2 = get_data_for_model(results, model2, datasets)

    skewness = skew(data_per_model_seen_datasets)
    kurt = kurtosis(data_per_model_unseen_datasets)  # Default is Fisher’s definition; add `fisher=False` for Pearson’s definition
    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurt}")
    #data_per_model_seen_datasets = data_per_model_unseen_datasets[:20]
    #data_per_model_unseen_datasets = data_per_model_unseen_datasets[21:]
    # print(data_per_model_seen_datasets)
    # print(data_per_model_unseen_datasets)
    print("group 1 length: ", len(data_per_model_seen_datasets))
    print("group 2 length: ", len(data_per_model_unseen_datasets))
    # p_value = perform_welchs_ttest(data_per_model_seen_datasets, data_per_model_unseen_datasets)
    # stat, p_value = mannwhitneyu(data_per_model_seen_datasets, data_per_model_unseen_datasets, alternative='two-sided')
    # print(f"U Statistic = {stat}, p-value = {p_value}")
    # Perform one-way ANOVA test
    p_values[model][metric] = p_value

    data_per_model_all = data_per_model_unseen_datasets + data_per_model_seen_datasets
    print(f"{model} {metric}, mean {np.mean(data_per_model_all)}, std {np.std(data_per_model_all)}")
    return p_values

# Function to perform Welch's t-test
def perform_welchs_ttest(group1, group2):
    # Welch's t-test doesn't assume equal variances
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    return p_value

# Base path where your text files are located
base_path = "D:\\Project\\seg2med_Project\\synthrad_results\\results_all_eval\\val_logs"  # Replace with the correct path
models = ["unet_ct", "cyclegan_ct", "pix2pix_ct", "ddpm_ct"]  # Replace with your model names
# Define a consistent color map for the models

#datasets = ["synthrad_256", "anish_256", "anika_256"]  # Replace with your dataset names
datasets = ["synthrad_512", "anish_512", "anika_512"] 

metrics = ["ssim", "psnr", "mae"]
output_dir = "D:\\Project\\seg2med_Project\\synthrad_results\\results_all_eval"
# Process and plot data
#results = load_and_process_data(base_path, models, datasets)
results = load_and_process_data_slices(base_path, models, datasets)
anonymized_results = anonymize_patient_ids(results)

'''data_per_dataset = {}
for model in models:
    #data_per_dataset[model] = {}  
    data_per_dataset[model] = [results[model][dataset].dropna() for dataset in datasets]

print(data_per_dataset["unet_ct"]["ssim"])'''



run_mode = "ddpm_greater"
if run_mode == "one_way_anova":
    # Run the ANOVA and print the results
    p_values = perform_anova(anonymized_results, datasets, metrics, models)

    for metric, p_value in p_values.items():
        print(f"P-value for {metric}: {p_value}")

elif run_mode == "two_group_anova":
    p_values = perform_anova_each_model(results, datasets, metrics, models)
    for model in p_values:
        for metric in p_values[model]:
            p_value = p_values[model][metric]
            print(f"P-value for {model} {metric}: {p_value}")
elif run_mode == "ddpm_greater":
    # Split metrics by model
    # print(anonymized_results['ddpm_ct'][datasets[0]]['ssim'].values.flatten().tolist())
    models2 = ['ddpm_ct', 'cyclegan_ct'] # unet_ct cyclegan_ct pix2pix_ct ddpm_ct
    target_metric = 'mae'
    target_metric_model1 = \
        anonymized_results[models2[0]][datasets[0]][target_metric].values.flatten().tolist()+\
        anonymized_results[models2[0]][datasets[1]][target_metric].values.flatten().tolist()+\
        anonymized_results[models2[0]][datasets[2]][target_metric].values.flatten().tolist()
        
    target_metric_model2 = \
        anonymized_results[models2[1]][datasets[0]][target_metric].values.flatten().tolist()+\
        anonymized_results[models2[1]][datasets[1]][target_metric].values.flatten().tolist()+\
        anonymized_results[models2[1]][datasets[2]][target_metric].values.flatten().tolist()
        
    print("length of target model 1:", len(target_metric_model1))
    print("length of target model 2:", len(target_metric_model2))
    # Perform one-tailed Mann-Whitney U Test for SSIM
    
    #stat, p = mannwhitneyu(target_metric_model1, target_metric_model2, alternative='greater') #'two-sided'\
    vis = False
    if vis:
        import matplotlib.pyplot as plt

        plt.hist(target_metric_model2, bins=20, edgecolor='k', alpha=0.7)
        plt.title("Histogram of Data")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()

    ktest=False
    if ktest:
        from scipy.stats import kstest

        stat, p = kstest(target_metric_model1, 'norm', args=(np.mean(target_metric_model1), np.std(target_metric_model1)))
        print(f"Statistic: {stat:.3f}, P-value: {p:.5f}")
        alpha = 0.05
        # Interpretation
        if p > alpha:
            print("Data appears to be normally distributed.")
        else:
            print("Data does not appear to be normally distributed.")

    mode = "utest"
    if mode == "ttest":
        stat, p = stats.ttest_ind(target_metric_model1, target_metric_model2, equal_var=True)
    elif mode == "utest":
        stat, p = mannwhitneyu(target_metric_model1, target_metric_model2, alternative='two-sided')
    print(f"Mann-Whitney U Test for {target_metric} p-value:", p)
    alpha = 0.05
    if p < alpha:
        print(f"Significant difference in SSIM between {models2[0]} and {models2[1]}.")
    else:
        print(f"No significant difference in SSIM between {models2[0]} and {models2[1]}.")
# ddpm vs unet Mann-Whitney U Test for SSIM p-value: 1.2222533820414355e-05
# 

def perform_tukey_hsd(results, metric, models):
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    # Flatten the data for Tukey's HSD
    data = []
    model_labels = []
    
    for model in models:
        data.extend(results[model][metric].dropna().tolist())
        model_labels.extend([model] * len(results[model][metric].dropna()))
        
    # Tukey's HSD test
    tukey_result = pairwise_tukeyhsd(data, model_labels, alpha=0.05)
    return tukey_result

# Perform Tukey's HSD for SSIM as an example
# tukey_ssim = perform_tukey_hsd(anonymized_results, "ssim", models)
# print(tukey_ssim)
