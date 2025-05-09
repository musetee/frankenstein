function [MAE] = Mean_Absolute_Error(ref,img)
% calculates the mean absolute error of an image from a reference image
MAE = sum(abs(img(img~=-2000)-ref(ref~=-2000)))/numel(ref(ref~=-2000));
end

