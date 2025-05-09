saveDirNM = [dataDirExperiment '\NM\'];
if ~exist(saveDirNM, 'dir')
    mkdir(saveDirNM)
end

synthetic_data_NM = cell(1, nrSynthPhantoms);
real_data_NM = cell(1, nrSynthPhantoms);

% 循环遍历每个phantom
for i = 1:nrSynthPhantoms
    % 读取合成数据
    synthData = nrrdread(fullfile(synthDir, synthNames{i}));
    
    % 读取相应的病人数据
    patData = nrrdread(fullfile(patientDir, patNames{i}));
    
    % 读取相应的肝脏掩膜
    maskData = nrrdread(fullfile(maskDir, maskNames{i}));
    maskData = double(maskData);  % 将掩膜转换为double类型以便一致运算
    organ_value = 5; % 5 for liver, 
    maskData = extractorganregion(maskData, organ_value);
    % 比较大小，如果不匹配发出警告
    [nrPatRows, nrPatColumns, nrPatSlices] = size(patData);
    [nrSynthRows, nrSynthColumns, nrSynthSlices] = size(synthData);
    
    if nrPatRows ~= nrSynthRows || nrPatColumns ~= nrSynthColumns
        warning('病人数据和合成数据的大小不匹配，将裁剪进行评估');
    end
    
    % 如果有必要，裁剪合成数据
    m = nrSynthRows - nrPatRows;
    n = nrSynthColumns - nrPatColumns;   
    synthData = synthData(1 + m/2 : end - m/2, 1 + n/2 : end - n/2, :); % 裁剪数据
    
    NMResultsSynth = zeros(nrSynthSlices, 1);
    NMResultsPat = zeros(nrSynthSlices, 1);

    