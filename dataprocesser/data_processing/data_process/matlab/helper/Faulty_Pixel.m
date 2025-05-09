function [FPR] = Faulty_Pixel(imageData)
% calculates the ratio of faulty pixel FPR in the background region of
% generated CT data, the backrgound region is determined using body contour
% data
totalPixel = numel(imageData(imageData~=-2000));
faultyPixel = numel(imageData(imageData > -900));
FPR = faultyPixel/totalPixel;
end

