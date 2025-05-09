function [FSIM, FSIMc] = FeatureSIM(imageRef, imageDis)
% calculate FSIM, replaced phasecong2 with phasecong3 (T.Russ)


% ========================================================================
% FSIM Index with automatic downsampling, Version 1.0
% Copyright(c) 2010 Lin ZHANG, Lei Zhang, Xuanqin Mou and David Zhang
% All Rights Reserved.
%
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
%
% This is an implementation of the algorithm for calculating the
% Feature SIMilarity (FSIM) index between two images.
%
% Please refer to the following paper
%
% Lin Zhang, Lei Zhang, Xuanqin Mou, and David Zhang,"FSIM: a feature similarity
% index for image qualtiy assessment", IEEE Transactions on Image Processing, vol. 20, no. 8, pp. 2378-2386, 2011.
% 
%----------------------------------------------------------------------
%
%Input : (1) imageRef: the first image being compared
%        (2) imageDis: the second image being compared
%
%Output: (1) FSIM: is the similarty score calculated using FSIM algorithm. FSIM
%	     only considers the luminance component of images. For colorful images, 
%            they will be converted to the grayscale at first.
%        (2) FSIMc: is the similarity score calculated using FSIMc algorithm. FSIMc
%            considers both the grayscale and the color information.
%Note: For grayscale images, the returned FSIM and FSIMc are the same.
%        
%-----------------------------------------------------------------------
%
%Usage:
%Given 2 test images img1 and img2. For gray-scale images, their dynamic range should be 0-255.
%For colorful images, the dynamic range of each color channel should be 0-255.
%
%[FSIM, FSIMc] = FeatureSIM(img1, img2);
%-----------------------------------------------------------------------

[rows, cols] = size(imageRef(:,:,1));
I1 = ones(rows, cols);
I2 = ones(rows, cols);
Q1 = ones(rows, cols);
Q2 = ones(rows, cols);

if ndims(imageRef) == 3 %images are colorful
    Y1 = 0.299 * double(imageRef(:,:,1)) + 0.587 * double(imageRef(:,:,2)) + 0.114 * double(imageRef(:,:,3));
    Y2 = 0.299 * double(imageDis(:,:,1)) + 0.587 * double(imageDis(:,:,2)) + 0.114 * double(imageDis(:,:,3));
    I1 = 0.596 * double(imageRef(:,:,1)) - 0.274 * double(imageRef(:,:,2)) - 0.322 * double(imageRef(:,:,3));
    I2 = 0.596 * double(imageDis(:,:,1)) - 0.274 * double(imageDis(:,:,2)) - 0.322 * double(imageDis(:,:,3));
    Q1 = 0.211 * double(imageRef(:,:,1)) - 0.523 * double(imageRef(:,:,2)) + 0.312 * double(imageRef(:,:,3));
    Q2 = 0.211 * double(imageDis(:,:,1)) - 0.523 * double(imageDis(:,:,2)) + 0.312 * double(imageDis(:,:,3));
else %images are grayscale
    Y1 = imageRef;
    Y2 = imageDis;
end

Y1 = double(Y1);
Y2 = double(Y2);
%%%%%%%%%%%%%%%%%%%%%%%%%
% Downsample the image
%%%%%%%%%%%%%%%%%%%%%%%%%
minDimension = min(rows,cols);
F = max(1,round(minDimension / 256));
aveKernel = fspecial('average',F);

aveI1 = conv2(I1, aveKernel,'same');
aveI2 = conv2(I2, aveKernel,'same');
I1 = aveI1(1:F:rows,1:F:cols);
I2 = aveI2(1:F:rows,1:F:cols);

aveQ1 = conv2(Q1, aveKernel,'same');
aveQ2 = conv2(Q2, aveKernel,'same');
Q1 = aveQ1(1:F:rows,1:F:cols);
Q2 = aveQ2(1:F:rows,1:F:cols);

aveY1 = conv2(Y1, aveKernel,'same');
aveY2 = conv2(Y2, aveKernel,'same');
Y1 = aveY1(1:F:rows,1:F:cols);
Y2 = aveY2(1:F:rows,1:F:cols);

%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the phase congruency maps
%%%%%%%%%%%%%%%%%%%%%%%%%
PC1 = phasecong3(Y1);
PC2 = phasecong3(Y2);

%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the gradient map
%%%%%%%%%%%%%%%%%%%%%%%%%
dx = [3 0 -3; 10 0 -10;  3  0 -3]/16;
dy = [3 10 3; 0  0   0; -3 -10 -3]/16;
IxY1 = conv2(Y1, dx, 'same');     
IyY1 = conv2(Y1, dy, 'same');    
gradientMap1 = sqrt(IxY1.^2 + IyY1.^2);

IxY2 = conv2(Y2, dx, 'same');     
IyY2 = conv2(Y2, dy, 'same');    
gradientMap2 = sqrt(IxY2.^2 + IyY2.^2);

%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the FSIM
%%%%%%%%%%%%%%%%%%%%%%%%%
T1 = 0.85;  %fixed
T2 = 160; %fixed
PCSimMatrix = (2 * PC1 .* PC2 + T1) ./ (PC1.^2 + PC2.^2 + T1);
gradientSimMatrix = (2*gradientMap1.*gradientMap2 + T2) ./(gradientMap1.^2 + gradientMap2.^2 + T2);
PCm = max(PC1, PC2);
SimMatrix = gradientSimMatrix .* PCSimMatrix .* PCm;
FSIM = sum(sum(SimMatrix)) / sum(sum(PCm));

%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the FSIMc
%%%%%%%%%%%%%%%%%%%%%%%%%
T3 = 200;
T4 = 200;
ISimMatrix = (2 * I1 .* I2 + T3) ./ (I1.^2 + I2.^2 + T3);
QSimMatrix = (2 * Q1 .* Q2 + T4) ./ (Q1.^2 + Q2.^2 + T4);

lambda = 0.03;

SimMatrixC = gradientSimMatrix .* PCSimMatrix .* real((ISimMatrix .* QSimMatrix) .^ lambda) .* PCm;
FSIMc = sum(sum(SimMatrixC)) / sum(sum(PCm));

return;









