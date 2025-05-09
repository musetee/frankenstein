clc
clear
pathes_predicted={'D:\Projects\SynthRad\logs\20231110_0256_Infer_DCGAN\saved_inference\mr\mr_Inference_valset_1.nii.gz';
    'D:\Projects\SynthRad\logs\20231110_0256_Infer_DCGAN\saved_inference\mr\mr_Inference_valset_2.nii.gz';
    'D:\Projects\SynthRad\logs\20231110_0256_Infer_DCGAN\saved_inference\mr\mr_Inference_valset_3.nii.gz';
    'D:\Projects\SynthRad\logs\20231110_0256_Infer_DCGAN\saved_inference\mr\mr_Inference_valset_4.nii.gz';
    'D:\Projects\SynthRad\logs\20231110_0256_Infer_DCGAN\saved_inference\mr\mr_Inference_valset_5.nii.gz';
    'D:\Projects\SynthRad\logs\20231110_0256_Infer_DCGAN\saved_inference\mr\mr_Inference_valset_6.nii.gz';
    'D:\Projects\SynthRad\logs\20231110_0256_Infer_DCGAN\saved_inference\mr\mr_Inference_valset_7.nii.gz';
    'D:\Projects\SynthRad\logs\20231110_0256_Infer_DCGAN\saved_inference\mr\mr_Inference_valset_8.nii.gz';
    'D:\Projects\SynthRad\logs\20231110_0256_Infer_DCGAN\saved_inference\mr\mr_Inference_valset_9.nii.gz';
    'D:\Projects\SynthRad\logs\20231110_0256_Infer_DCGAN\saved_inference\mr\mr_Inference_valset_10.nii.gz'};
pathes_real={
    'D:\Projects\data\Task1\pelvis\1PC082\ct.nii.gz';
    'D:\Projects\data\Task1\pelvis\1PC084\ct.nii.gz';
    'D:\Projects\data\Task1\pelvis\1PC085\ct.nii.gz';
    'D:\Projects\data\Task1\pelvis\1PC088\ct.nii.gz';
    'D:\Projects\data\Task1\pelvis\1PC092\ct.nii.gz';
    'D:\Projects\data\Task1\pelvis\1PC093\ct.nii.gz';
    'D:\Projects\data\Task1\pelvis\1PC095\ct.nii.gz';
    'D:\Projects\data\Task1\pelvis\1PC096\ct.nii.gz';
    'D:\Projects\data\Task1\pelvis\1PC097\ct.nii.gz';
    'D:\Projects\data\Task1\pelvis\1PC098\ct.nii.gz'};
all_ssim = [];
all_psnr = [];
all_mae = [];
for i = 1:length(pathes_real)
    [ssim, mae, psnr] = evaluateMetrics(pathes_real{i}, pathes_predicted{i});
    all_ssim=[all_ssim;ssim];
    all_mae=[all_mae;mae];
    all_psnr=[all_psnr;psnr];
end

mean_ssim=mean(all_ssim);
mean_mae=mean(all_mae);
mean_psnr=mean(all_psnr);

std_ssim=std(all_ssim);
std_mae=std(all_mae);
std_psnr=std(all_psnr);

