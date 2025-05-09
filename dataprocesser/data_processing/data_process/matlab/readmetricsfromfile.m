% Define the file path to your text file
file_path = 'D:\Projects\Diffusion-Models-pytorch\logs\results\images\20231028_1516_Infer_Unet\saved_logs\infer_all_log.txt';

% Read the entire file
file_contents = fileread(file_path);

% Define regular expressions to match the lines with "val set" and the metrics
pattern_val_set = 'val set \d+,\s+SSIM: metatensor\(([\d.]+)\),\s+MAE: metatensor\(([\d.]+)\),\s+PSNR: metatensor\(([\d.]+)\)';
pattern_overall_metrics = 'over all metrices,\s+SSIM: metatensor\(([\d.]+)\),\s+MAE: metatensor\(([\d.]+)\),\s+PSNR: metatensor\(([\d.]+)\)';

% Use regular expressions to extract values from the file
val_set_matches = regexp(file_contents, pattern_val_set, 'tokens');
overall_metrics_matches = regexp(file_contents, pattern_overall_metrics, 'tokens');

% Initialize arrays to store the extracted values
SSIM_val_set = [];
MAE_val_set = [];
PSNR_val_set = [];

% Loop through val set matches and extract the values
for i = 1:numel(val_set_matches)
    values = val_set_matches{i};
    SSIM_val_set(i) = str2double(values{1});
    MAE_val_set(i) = str2double(values{2});
    PSNR_val_set(i) = str2double(values{3});
end

% Display the extracted values for "val set" lines
% disp('SSIM (val set):');
% disp(SSIM_val_set);
% 
% disp('MAE (val set):');
% disp(MAE_val_set);
% 
% disp('PSNR (val set):');
% disp(PSNR_val_set);
mean_SSIM = mean(SSIM_val_set);
mean_MAE = mean(MAE_val_set);
mean_PSNR = mean(PSNR_val_set);

std_SSIM = std(SSIM_val_set);
std_MAE = std(MAE_val_set);
std_PSNR = std(PSNR_val_set);

