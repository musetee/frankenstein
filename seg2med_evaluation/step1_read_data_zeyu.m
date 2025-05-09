% Specify the folder containing the files
clear
root = 'D:\Project\seg2med_Project\SynthRad_GAN\seg2med_evaluation';
dataDir = [root, '\MedicalImageEvaluation\'];
experiment = 'Results1201_synthetic';

% XCATCT
% folder_path = 'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase2_xcat_synthetic\20241119_0028_Infer_ddpm2d_seg2med_XCAT_CT_56Models_64slices_512\saved_outputs\volume_output';

% Anish
folder_path = 'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase1_ct_synthrad_anish_4_models\20241119_1142_Infer_ddpm2d_seg2med_anish_512\saved_outputs\volume_output';

% Anika
% folder_path = 'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase1_ct_anika_4_models\Infer_ddpm2d_seg2med_anika_512_all\saved_outputs\volume_output';

% Synthetic
folder_path = 'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase2_xcat_synthetic\20241120_2348_Infer_ddpm2d_seg2med_synthetic_512_ct\saved_outputs\volume_output';
synthDir = folder_path;
patientDir = folder_path;
maskDir = folder_path;
% Get a list of all .nrrd files in the folder
files = dir(fullfile(folder_path, '*.nii.gz'));

% Initialize arrays to store grouped file paths for each category
maskList = {};
segList = {};
synthList = {};
patientList = {};

% Loop through each file
for i = 1:length(files)
    % Get the full file name
    file_name = files(i).name;
    %file_path = fullfile(folder_path, file_name);
    
    % Extract patient ID and type using regular expressions
    tokens = regexp(file_name, '^(\w+)_(\w+)_volume\.nii.gz$', 'tokens');
    
    % Check if the file name matches the expected pattern
    if ~isempty(tokens)
        % Extract volume type (e.g., 'mask', 'seg', 'synthesized', 'target')
        volume_type = tokens{1}{2};
        
        % Add the file path to the corresponding category based on volume type
        switch volume_type
            case 'mask'
                maskList{end+1} = file_name;
            case 'seg'
                segList{end+1} = file_name;
            case 'synthesized'
                synthList{end+1} = file_name;
            case 'target'
                patientList{end+1} = file_name;
        end
    end
end

nRead = 1;
nrSynthPhantoms = nRead;
% 读取文件
synthNames = synthList(1:nRead); % 只取前10个文件
patNames = patientList(1:nRead); % 只取前10个文件
maskNames = maskList(1:nRead); % 只取前10个文件

% Display results (optional)
disp('Mask Files:'), disp(maskNames)
disp('Synthesized Files:'), disp(synthNames)
disp('Target Files:'), disp(patNames)
