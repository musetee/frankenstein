clc
clear
realImageNifti = niftiread('D:\Projects\data\Task1\pelvis\1PC098\ct.nii.gz');
predictedImageNifti= niftiread('D:\Projects\SynthRad\logs\20231110_0256_Infer_DCGAN\saved_inference\mr\mr_Inference_valset_10.nii.gz');

idx=50;
realImage = realImageNifti(:,:,idx);
predictedImage = predictedImageNifti(:,:,idx);

[ssimval, ssimmap] = ssim(realImage, predictedImage, 'DynamicRange', 3000);
maeValue = mean(abs(double(predictedImage) - double(realImage)), 'all');
psnrValue = psnr(predictedImage, realImage, 3000);