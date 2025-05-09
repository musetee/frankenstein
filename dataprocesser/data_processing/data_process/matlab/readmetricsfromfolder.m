clear
% Define a folder containing your text files
folder_path = 'D:\Projects\Diffusion-Models-pytorch\logs\results\logs\20231028_1753_Infer_monai_pix2pix\saved_logs';

% Create arrays to store the metric values
all_SSIM = [];
all_MAE = [];
all_PSNR = [];

% Get a list of all text files in the folder
file_list = dir(fullfile(folder_path, '*.txt'));

for i = 1:length(file_list)
    file_name = file_list(i).name;
    
    % Check if the file name contains "infer_log_step"
    if contains(file_name, 'infer_log_step')
        file_path = fullfile(folder_path, file_name);
        
        % Read the file
        file_contents = fileread(file_path);
        
        % -?
        % Define regular expressions to match the lines with "val set" and the metrics
        pattern_val_set = 'val set \d+,\s+SSIM: metatensor\((-?[\d.]+)\),\s+MAE: metatensor\((-?[\d.]+)\),\s+PSNR: metatensor\((-?[\d.]+)\)';
        
        % Use regular expressions to extract values from the file
        val_set_matches = regexp(file_contents, pattern_val_set, 'tokens');
        
        % Loop through val set matches and extract the values
        for j = 1:numel(val_set_matches)
            values = val_set_matches{j};
            SSIM_val = str2double(values{1});
            MAE_val = str2double(values{2});
            PSNR_val = str2double(values{3});
            
            % Append values to the arrays
            all_SSIM = [all_SSIM, SSIM_val];
            all_MAE = [all_MAE, MAE_val];
            all_PSNR = [all_PSNR, PSNR_val];
        end
    end
end

% Calculate mean and standard deviation for each metric
mean_SSIM = mean(all_SSIM);
std_SSIM = std(all_SSIM);

mean_MAE = mean(all_MAE);
std_MAE = std(all_MAE);

mean_PSNR = mean(all_PSNR);
std_PSNR = std(all_PSNR);

% Display the results
fprintf('Mean SSIM: %f, Std SSIM: %f\n', mean_SSIM, std_SSIM);
fprintf('Mean MAE: %f, Std MAE: %f\n', mean_MAE, std_MAE);
fprintf('Mean PSNR: %f, Std PSNR: %f\n', mean_PSNR, std_PSNR);