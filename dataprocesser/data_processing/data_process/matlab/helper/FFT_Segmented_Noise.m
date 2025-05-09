function [NPS2D,ROIs,ftROIs] = FFT_Segmented_Noise(segmNoise)
% calculates 2D NPS from segmented noise realizations
ftROIs = zeros(size(segmNoise));
ROIs = zeros(size(segmNoise));
nonZeroFtROIs = [];
for z = 1:size(segmNoise,3)
    ROI = segmNoise(:,:,z);
    gaussROI = imgaussfilt(ROI,2);
    ROI = ROI - gaussROI; % subtract the gauss filtered image to remove low frequency structures due to segmentation
    ROIs(:,:,z) = ROI;
    nrPixels = numel(nonzeros(gaussROI));
    ftROI = fftshift(fft2(ROI));
    absftROI = abs(ftROI).^2;
    absftROI = absftROI./nrPixels;
    ftROIs(:,:,z) = absftROI;
    if nrPixels ~= 0
        nonZeroFtROIs(:,:,end+1) = absftROI;       
    end
end
nonZeroFtROIs(:,:,1) = [];
NPS2D = mean(nonZeroFtROIs,3);
end

