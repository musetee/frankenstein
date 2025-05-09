Write-Output "Activating virtual environment"
D:\Projects\Environments\torch\venv\Scripts\activate.ps1
Write-Output "Running spacing.py"
$mode0 = "pred"
python spacing.py --mode $mode0 --axis 2 --reference_image D:\Projects\data\Task1\pelvis\1PC082\ct.nii.gz --image .\logs\ddpm_200_results\1PC082test.nii.gz
python spacing.py --mode $mode0 --axis 2 --reference_image D:\Projects\data\Task1\pelvis\1PC092\ct.nii.gz --image .\logs\ddpm_200_results\1PC084test.nii.gz
python spacing.py --mode $mode0 --axis 2 --reference_image D:\Projects\data\Task1\pelvis\1PC093\ct.nii.gz --image .\logs\ddpm_200_results\1PC085test.nii.gz
python spacing.py --mode $mode0 --axis 2 --reference_image D:\Projects\data\Task1\pelvis\1PC096\ct.nii.gz --image .\logs\ddpm_200_results\1PC088test.nii.gz
python spacing.py --mode $mode0 --axis 2 --reference_image D:\Projects\data\Task1\pelvis\1PC097\ct.nii.gz --image .\logs\ddpm_200_results\1PC092test.nii.gz
python spacing.py --mode $mode0 --axis 2 --reference_image D:\Projects\data\Task1\pelvis\1PC098\ct.nii.gz --image .\logs\ddpm_200_results\1PC093test.nii.gz
python spacing.py --mode $mode0 --axis 2 --reference_image D:\Projects\data\Task1\pelvis\1PC098\ct.nii.gz --image .\logs\ddpm_200_results\1PC095test.nii.gz
python spacing.py --mode $mode0 --axis 2 --reference_image D:\Projects\data\Task1\pelvis\1PC098\ct.nii.gz --image .\logs\ddpm_200_results\1PC096test.nii.gz
python spacing.py --mode $mode0 --axis 2 --reference_image D:\Projects\data\Task1\pelvis\1PC098\ct.nii.gz --image .\logs\ddpm_200_results\1PC097test.nii.gz
python spacing.py --mode $mode0 --axis 2 --reference_image D:\Projects\data\Task1\pelvis\1PC098\ct.nii.gz --image .\logs\ddpm_200_results\1PC098test.nii.gz