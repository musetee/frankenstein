function [mean_ssimval, mean_maeValue, mean_psnrValue] = evaluateMetrics(path_real, path_predicted)
% 'D:\Projects\SynthRad\logs\20231110_0256_Infer_DCGAN\saved_inference\mr\mr_Inference_valset_7.nii.gz'
% 'D:\Projects\data\Task1\pelvis\1PC095\ct.nii.gz'
realImageNifti = niftiread(path_real);
predictedImageNifti= niftiread(path_predicted);

idx=50;
realImage = realImageNifti(:,:,idx);
predictedImage = predictedImageNifti(:,:,idx);

[ssimval, ssimmap] = ssim(realImage, predictedImage, 'DynamicRange', 3000);
maeValue = mean(abs(double(predictedImage) - double(realImage)), 'all');
psnrValue = psnr(predictedImage, realImage, 3000);


[mean_ssimval, mean_ssimmap] = ssim(realImageNifti, predictedImageNifti, 'DynamicRange', 3000);
mean_maeValue = mean(abs(double(predictedImageNifti) - double(realImageNifti)), 'all');
mean_psnrValue = psnr(predictedImageNifti, realImageNifti, 3000);
mean_ssimval = abs(mean_ssimval);

% % display one image
% figure; % Creates a new figure
% imagesc(predictedImage); % Display the image
% colormap gray; % Set colormap to gray for grayscale image
% axis image; % Adjust the axis to image aspect ratio
% colorbar; % Optional: shows a colorbar
% 
% figure; % Creates a new figure
% imagesc(realImage); % Display the image
% colormap gray; % Set colormap to gray for grayscale image
% axis image; % Adjust the axis to image aspect ratio
% colorbar; % Optional: shows a colorbar