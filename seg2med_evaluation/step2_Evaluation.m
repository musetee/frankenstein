% 添加必要的路径
% addpath(genpath('E:\XCATproject\Eval_test\helper'));
clear

step1_read_data_zeyu
organ_value = 1; % 5 for liver, 

% 选择不同指标的保存目录

dataDirExperiment = [dataDir experiment];
saveDirEPR_EGR = [dataDirExperiment '\EPR_EGR\'];
saveDirNCC = [dataDirExperiment '\NCC\'];
saveDirHistCC = [dataDirExperiment '\HistCC\'];
saveDirFSIM = [dataDirExperiment '\FSIM\'];

% 如果目录不存在，创建目录
if ~exist(saveDirEPR_EGR, 'dir')
    mkdir(saveDirEPR_EGR)
end

if ~exist(saveDirNCC, 'dir')
    mkdir(saveDirNCC)
end

if ~exist(saveDirHistCC, 'dir')
    mkdir(saveDirHistCC)
end

if ~exist(saveDirFSIM, 'dir')
    mkdir(saveDirFSIM)
end

saveDirNM = [dataDirExperiment '\NM\']; %NMThings
if ~exist(saveDirNM, 'dir')
    mkdir(saveDirNM)
end


% 设置直方图的边缘（bins）
edges = 0:1:255; % 根据图像强度范围进行调整

% 初始化存储用于箱线图的数据的cell数组
EPR_data = cell(1, nrSynthPhantoms);
EGR_data = cell(1, nrSynthPhantoms);
NCC_data = cell(1, nrSynthPhantoms);
HistCC_data = cell(1, nrSynthPhantoms);
FSIM_data = cell(1, nrSynthPhantoms);
synthetic_data_NM = cell(1, nrSynthPhantoms); %NMThings
real_data_NM = cell(1, nrSynthPhantoms); %NMThings

% 循环遍历每个phantom
for i = 1:nrSynthPhantoms
    if endsWith(synthNames{i}, '.nrrd')
        % 读取合成数据
        synthData = nrrdread(fullfile(synthDir, synthNames{i}));
        
        % 读取相应的病人数据
        patData = nrrdread(fullfile(patientDir, patNames{i}));
        
        % 读取相应的肝脏掩膜
        maskData = nrrdread(fullfile(maskDir, maskNames{i}));
    elseif endsWith(synthNames{i}, '.nii.gz')
        % 读取合成数据
        synthData = load_nii(fullfile(synthDir, synthNames{i})).img;
        
        % 读取相应的病人数据
        patData = load_nii(fullfile(patientDir, patNames{i})).img;
        
        % 读取相应的肝脏掩膜
        maskData = load_nii(fullfile(maskDir, maskNames{i})).img;
    end
    maskData = double(maskData);  % 将掩膜转换为double类型以便一致运算
    
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
    
    % 初始化存储每个切片的结果
    [EPRtmp, EGRtmp] = deal(zeros(nrSynthSlices, 1));
    NMResultsSynth = zeros(nrSynthSlices, 1);
    NMResultsPat = zeros(nrSynthSlices, 1);
    NCCResults = zeros(nrSynthSlices, 1);
    histCCResults = zeros(nrSynthSlices, 1);
    FSIMResults = zeros(nrSynthSlices, 1);
    
    % 循环遍历每个切片
    for j = 1:nrSynthSlices
        % 获取当前切片的合成和病人数据
        synthSlice = synthData(:, :, j);
        patSlice = patData(:, :, j);
        maskSlice = maskData(:, :, j);
        
        % 对EPR, EGR和FSIM计算进行重缩放
        synthSliceRescaled = rescale(synthSlice, 0, 255);
        patSliceRescaled = rescale(patSlice, 0, 255);
        
        % 计算EPR和EGR
        [EPRtmp(j), EGRtmp(j)] = Edge_Preservation_Ratio(patSliceRescaled, synthSliceRescaled);
        
        % 对合成和病人数据应用肝脏掩膜以计算NM
        synthSliceMasked = double(synthSlice) .* maskSlice;
        patSliceMasked = double(patSlice) .* maskSlice;
        
        % 计算合成和病人切片的NM（噪声量度）
        if isnan(std(synthSliceMasked(maskSlice > 0)))
            NMResultsSynth(j) = 0;
            NMResultsPat(j) = 0;
        else
            NMResultsSynth(j) = std(synthSliceMasked(maskSlice > 0));
            NMResultsPat(j) = std(patSliceMasked(maskSlice > 0));
        end
        
        % 计算NCC
        if all(synthSliceMasked(:) == 0) || all(patSliceMasked(:) == 0)
            NCCResults(j) = NaN;
        else
            synthNPS = FFT_Segmented_Noise(synthSliceMasked);
            patNPS = FFT_Segmented_Noise(patSliceMasked);
            pixelSpacing = 1; % 根据图像元数据进行调整
            [~, synthRadialNPS] = Radial_from_2D_NPS(synthNPS, pixelSpacing, 1);
            [~, patRadialNPS] = Radial_from_2D_NPS(patNPS, pixelSpacing, 1);
            tmp = corrcoef(synthRadialNPS, patRadialNPS);
            NCCResults(j) = tmp(1, 2);
        end
        
        % 计算直方图相关性（HistCC）
        synthHist = histcounts(synthSlice(:), edges) / numel(synthSlice);
        patHist = histcounts(patSlice(:), edges) / numel(patSlice);
        tmp = corrcoef(synthHist, patHist);
        histCCResults(j) = tmp(1, 2);

        % 计算FSIM
        [FSIMResults(j), ~] = FeatureSIM(patSliceRescaled, synthSliceRescaled);
    end
    
    % 存储用于箱线图的数据
    EPR_data{i} = EPRtmp(~isnan(EPRtmp));
    EGR_data{i} = EGRtmp(~isnan(EGRtmp));
    synthetic_data_NM{i} = NMResultsSynth(~isnan(NMResultsSynth));
    real_data_NM{i} = NMResultsPat(~isnan(NMResultsPat));
    NCC_data{i} = NCCResults(~isnan(NCCResults));
    HistCC_data{i} = histCCResults(~isnan(histCCResults));
    FSIM_data{i} = FSIMResults(~isnan(FSIMResults));

    % 将当前phantom的结果保存为CSV文件
    [~, patFileName, ~] = fileparts(patNames{i});
    validFileName = matlab.lang.makeValidName(patFileName);

    % 保存EPR和EGR结果
    resultTableEPR_EGR = table(EPRtmp, EGRtmp, 'VariableNames', {'EPR', 'EGR'});
    writetable(resultTableEPR_EGR, fullfile(saveDirEPR_EGR, [validFileName '_EPR_EGR.csv']));

    % 保存NM结果
    resultTableNM = table((1:nrSynthSlices)', NMResultsSynth, NMResultsPat, 'VariableNames', {'Slice', 'NM_Synth', 'NM_Pat'});
    writetable(resultTableNM, fullfile(saveDirNM, [validFileName '_NM.csv']));

    % 保存NCC结果
    resultTableNCC = table((1:nrSynthSlices)', NCCResults, 'VariableNames', {'Slice', 'NCC'});
    writetable(resultTableNCC, fullfile(saveDirNCC, [validFileName '_NCC.csv']));

    % 保存HistCC结果
    resultTableHistCC = table((1:nrSynthSlices)', histCCResults, 'VariableNames', {'Slice', 'HistCC'});
    writetable(resultTableHistCC, fullfile(saveDirHistCC, [validFileName '_HistCC.csv']));

    % 保存FSIM结果
    resultTableFSIM = table(FSIMResults, 'VariableNames', {'FSIM'});
    writetable(resultTableFSIM, fullfile(saveDirFSIM, [validFileName '_FSIM.csv']));
end

disp('所有计算和CSV文件保存完成.');

%% 计算并输出所有组数据的平均值和标准差
% 计算EPR
all_EPR_data = vertcat(EPR_data{:});
mean_EPR = mean(all_EPR_data);
std_EPR = std(all_EPR_data);
fprintf('EPR 数据: 平均值 = %.4f, 标准差 = %.4f\n', mean_EPR, std_EPR);

% 计算EGR
all_EGR_data = vertcat(EGR_data{:});
mean_EGR = mean(all_EGR_data);
std_EGR = std(all_EGR_data);
fprintf('EGR 数据: 平均值 = %.4f, 标准差 = %.4f\n', mean_EGR, std_EGR);

% 计算NM
all_synthetic_NM = vertcat(synthetic_data_NM{:});
all_real_NM = vertcat(real_data_NM{:});
mean_synthetic_NM = mean(all_synthetic_NM);
std_synthetic_NM = std(all_synthetic_NM);
mean_real_NM = mean(all_real_NM);
std_real_NM = std(all_real_NM);
fprintf('合成图像 NM 数据: 平均值 = %.4f, 标准差 = %.4f\n', mean_synthetic_NM, std_synthetic_NM);
fprintf('真实图像 NM 数据: 平均值 = %.4f, 标准差 = %.4f\n', mean_real_NM, std_real_NM);

% 计算NCC
all_NCC_data = vertcat(NCC_data{:});
mean_NCC = mean(all_NCC_data);
std_NCC = std(all_NCC_data);
fprintf('NCC 数据: 平均值 = %.4f, 标准差 = %.4f\n', mean_NCC, std_NCC);

% 计算HistCC
all_HistCC_data = vertcat(HistCC_data{:});
mean_HistCC = mean(all_HistCC_data);
std_HistCC = std(all_HistCC_data);
fprintf('HistCC 数据: 平均值 = %.4f, 标准差 = %.4f\n', mean_HistCC, std_HistCC);

% 计算FSIM
all_FSIM_data = vertcat(FSIM_data{:});
mean_FSIM = mean(all_FSIM_data);
std_FSIM = std(all_FSIM_data);
fprintf('FSIM 数据: 平均值 = %.4f, 标准差 = %.4f\n', mean_FSIM, std_FSIM);



%% 绘制并保存EPR箱线图
plot_and_save_boxplot(EPR_data, 'EPR Value', 'EPR_BoxPlot.png', dataDirExperiment);

%% 绘制并保存EGR箱线图
plot_and_save_boxplot(EGR_data, 'EGR Value', 'EGR_BoxPlot.png', dataDirExperiment);

%% 绘制并保存NM箱线图
plot_and_save_nm_boxplot(synthetic_data_NM, real_data_NM, 'NM_BoxPlot.png', dataDirExperiment);

%% 绘制并保存NCC箱线图
plot_and_save_boxplot(NCC_data, 'NCC Value', 'NCC_BoxPlot.png', dataDirExperiment);

%% 绘制并保存HistCC箱线图
plot_and_save_boxplot(HistCC_data, 'HistCC Value', 'HistCC_BoxPlot.png', dataDirExperiment);

%% 绘制并保存FSIM箱线图
plot_and_save_boxplot(FSIM_data, 'FSIM Value', 'FSIM_BoxPlot.png', dataDirExperiment);

%% 辅助函数：通用箱线图绘制函数
function plot_and_save_boxplot(data, ylabel_text, filename, dataDir)
    % 设置箱线图颜色
    color = [106/255, 90/255, 205/255];
    
    % 创建图形窗口
    figure;

    % 设置箱线图位置
    group_spacing = 5; % 组之间的间隔
    positions = 1:group_spacing:(length(data) * group_spacing);

    % 循环绘制每组的箱线图
    for i = 1:length(data)
        boxplot(data{i}, 'Colors', color, 'symbol', '', 'positions', positions(i), 'Widths', 0.4);
        hold on;
    end

    % 设置X轴标签
    set(gca, 'XTick', positions);
    set(gca, 'XTickLabel', arrayfun(@(x) num2str(x), 1:length(data), 'UniformOutput', false));

    % 设置其他图形属性
    set(gca,'FontName', 'Times New Roman', 'FontSize', 18);
    ylabel(ylabel_text);
    xlabel('Testset Number');
    set(gca, 'linewidth', 1.5);

    % 设置Y轴范围（可以根据需要调整）
    ylim([0, max(cellfun(@max, data)) * 1.2]);

    % 设置X轴的范围，增加左右边距
    xlim([min(positions)-group_spacing, max(positions)+group_spacing]);

    % 设置箱线图的线条宽度
    set(findobj(gca,'type','line'),'linewidth',1.2); % 线条宽度

    % 显示网格线
    grid off;

    hold off;

    % 保存箱线图
    saveas(gcf, fullfile(dataDir, filename));
end

%% 辅助函数：NM箱线图绘制函数（两个数据集）
function plot_and_save_nm_boxplot(synthetic_data, real_data, filename, dataDir)
    % 设置箱线图颜色
    colors = [1.0, 0.498, 0.447; 106/255, 90/255, 205/255];

    % 创建图形窗口
    figure;

    % 设置箱线图位置
    group_spacing = 5; % 组之间的间隔
    inner_spacing = 1; % 组内间隔

    positions = [];
    for i = 1:length(synthetic_data)
        positions = [positions, i*group_spacing + (i-1)*inner_spacing, i*group_spacing + i*inner_spacing];
    end

    % 循环绘制每组的箱线图
    for i = 1:length(synthetic_data)
        boxplot([synthetic_data{i}, real_data{i}], 'Colors', colors, ...
                'symbol', '', 'positions', positions(2*i-1:2*i), 'Widths', 0.4);
        hold on;
    end

    % 设置X轴标签
    set(gca, 'XTick', mean(reshape(positions, 2, [])', 2));
    set(gca, 'XTickLabel', arrayfun(@(x) num2str(x), 1:length(synthetic_data), 'UniformOutput', false));

    % 设置图例
    h = findobj(gca,'Tag','Box');
    legend(h([2, 1]), {'Synthetic Image NM', 'Real Image NM'}, 'Location', 'northeast');

    % 设置其他图形属性
    set(gca,'FontName', 'Times New Roman', 'FontSize', 18);
    ylabel('NM Value');
    xlabel('Testset Number');
    set(gca, 'linewidth', 1.5);

    % 设置X轴的范围，增加左右边距
    xlim([min(positions)-group_spacing, max(positions)+group_spacing]);

    % 设置箱线图的线条宽度
    set(findobj(gca,'type','line'),'linewidth',1.2); % 线条宽度

    % 显示网格线
    grid off;

    hold off;

    % 保存箱线图
    saveas(gcf, fullfile(dataDir, filename));
end
