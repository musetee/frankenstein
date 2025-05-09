function [dist,avg] = Radial_Average(A,acc)
%function to calculate the radial average of a quadratic matrix A with accuracy bins of size acc
dataSize = size(A);
if mod(dataSize(1),2) == 1
    x = -floor(dataSize(1)/2):floor(dataSize(1)/2);
elseif mod(dataSize(1),2) == 0
    x = -dataSize(1)/2+0.5:dataSize(1)/2-0.5;
end
if mod(dataSize(2),2) == 1
    y = -floor(dataSize(2)/2):floor(dataSize(2)/2);
elseif mod(dataSize(2),2) == 0
    y = -dataSize(2)/2+0.5:dataSize(2)/2-0.5;
end
[X,Y] = meshgrid(x,y);
R = sqrt(X.^2+Y.^2);
% rebin array values
R = round(R/acc)*acc;
% average over array values with same bin
dist = unique(R)';
if mod(dataSize(1),2) == 1
    dist(dist>(dataSize(1)-1)/2) = [];
elseif mod(dataSize(2),2) == 0
    dist(dist>dataSize(1)/2) = [];
end
avg = [];
for idx = 1:numel(dist)
    value = dist(idx);
    avg(idx) = mean(A(abs(R-value) < 1e4*eps(min(abs(R),abs(value))))); % check wheter R == value and average over corresponding values
end
end