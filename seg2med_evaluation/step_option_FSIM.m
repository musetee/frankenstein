% Adding necessary paths and clearing the workspace
clear
step1_read_data_zeyu
organ_value = 1; % Specify organ value, e.g., 5 for liver

% Set the directory to save FSIM results
dataDirExperiment = [dataDir experiment];
saveDirFSIM = [dataDirExperiment '\FSIM\'];

% Create directory if it doesn't exist
if ~exist(saveDirFSIM, 'dir')
    mkdir(saveDirFSIM)
end

% Loop through each phantom
for i = 1:nrSynthPhantoms
    if endsWith(synthNames{i}, '.nrrd')
        % Read synthetic data
        synthData = nrrdread(fullfile(synthDir, synthNames{i}));
        % Read corresponding ground truth data
        patData = nrrdread(fullfile(patientDir, patNames{i}));
    elseif endsWith(synthNames{i}, '.nii.gz')
        % Read synthetic data
        synthData = load_nii(fullfile(synthDir, synthNames{i})).img;
        % Read corresponding ground truth data
        patData = load_nii(fullfile(patientDir, patNames{i})).img;
    end

    % Ensure mask data matches size and extract the specified organ region
    maskData = nrrdread(fullfile(maskDir, maskNames{i}));
    maskData = double(maskData);
    maskData = extractorganregion(maskData, organ_value);

    % Adjust dimensions if necessary
    [nrPatRows, nrPatColumns, nrPatSlices] = size(patData);
    [nrSynthRows, nrSynthColumns, ~] = size(synthData);

    if nrPatRows ~= nrSynthRows || nrPatColumns ~= nrSynthColumns
        warning('Mismatch in dimensions. Cropping data for evaluation.');
        m = nrSynthRows - nrPatRows;
        n = nrSynthColumns - nrPatColumns;
        synthData = synthData(1 + m/2 : end - m/2, 1 + n/2 : end - n/2, :);
    end

    % Initialize FSIM results
    FSIMResults = zeros(nrSynthSlices, 1);

    % Loop through each slice to compute FSIM
    for j = 1:nrSynthSlices
        % Extract slices from synthetic and ground truth images
        synthSlice = synthData(:, :, j);
        patSlice = patData(:, :, j);

        % Rescale slices for FSIM calculation
        synthSliceRescaled = rescale(synthSlice, 0, 255);
        patSliceRescaled = rescale(patSlice, 0, 255);

        % Calculate FSIM for the current slice
        [FSIMResults(j), ~] = FeatureSIM(patSliceRescaled, synthSliceRescaled);
    end

    % Save FSIM results for the current phantom
    [~, patFileName, ~] = fileparts(patNames{i});
    validFileName = matlab.lang.makeValidName(patFileName);

    resultTableFSIM = table(FSIMResults, 'VariableNames', {'FSIM'});
    writetable(resultTableFSIM, fullfile(saveDirFSIM, [validFileName '_FSIM.csv']));
end

disp('FSIM calculation completed and results saved.');
