nrSynthPhantoms% 初始化参数
root = 'D:\Project\zhilin_master\final\Evaluation';
dataDir = [root, '\MedicalImageEvaluation\'];

% 准备数据读取路径
synthDir = [dataDir 'synthDir_processed\']; % 生成的合成图像目录
patientDir = [dataDir 'patientDir\']; % 原始病人图像目录
maskDir = [dataDir 'maskDir\']; % 肝脏掩膜图像目录

% 读取文件
synthList = dir(fullfile(synthDir, '*.nrrd'));
synthNames = {synthList(1:10).name}; % 只取前10个文件
nrSynthPhantoms = numel(synthNames);

patientList = dir(fullfile(patientDir, '*.nrrd'));
patNames = {patientList(1:10).name}; % 只取前10个文件

maskList = dir(fullfile(maskDir, '*.nrrd'));
maskNames = {maskList(1:10).name}; % 只取前10个文件