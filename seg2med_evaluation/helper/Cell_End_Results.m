function [result] = Cell_End_Results(res)
% calculates mean and std of cell array data and writes it in an Latex
% readable form: mean +- std
% res = cell2mat(res);
resLength = size(res,2);
result = cell(resLength,1);
for i = 1:resLength
    resMean = nanmean(vertcat(res{:,i}),1);
    resSTD = nanstd(vertcat(res{:,i}),0,1);
    result{i} = sprintf('%.3f $\\pm$ %.3f',resMean,resSTD);
end
end

