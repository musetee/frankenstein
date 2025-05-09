function [freq,radNPS] = Radial_from_2D_NPS(NPS2D,pixelSpacing,acc)
% calculates the radial NPS by radial averaging and the radial frequency
% input: 2DNPS, PixelSpacing, and accuracy for rounding
sampling = 1/pixelSpacing; % sampling rate [1/mm] for quadratic image
Nyquist = sampling/2;
[RFreq,radNPS] = Radial_Average(NPS2D,acc);
freq = linspace(0,Nyquist,size(RFreq,2))';
end

