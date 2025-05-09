function [c_m] = Calc_Confusion_Matrix(XCAT,synth,maxVal,minVal)
%% Load Data
XCAT = int16(XCAT);
synth = int16(synth);
offset = 1 - minVal;
n_v = abs(maxVal - minVal)+1;
%% Generate Confusion Matrix
c_m = zeros(n_v,n_v);
[s_x,s_y,s_z] = size(XCAT);
for x = 1:s_x
    for y = 1:s_y
        for z = 1:s_z
            i = XCAT(x,y,z) + offset;
            j = synth(x,y,z) + offset;
            c_m(i,j) = c_m(i,j)+1;
        end
    end
end
end