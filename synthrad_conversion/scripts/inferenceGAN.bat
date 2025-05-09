@echo off

:: Activate the virtual environment
F:\yang_Environments\torch\venv\Scripts\activate.ps1

python F:\yang_Projects\SynthRad_GAN\train_2d.py --config ./configs/newserver/inference_unet.yaml
python F:\yang_Projects\SynthRad_GAN\train_2d.py --config ./configs/inference_attunet.yaml
python F:\yang_Projects\SynthRad_GAN\train_2d.py --config ./configs/inference_dcgan.yaml
python F:\yang_Projects\SynthRad_GAN\train_2d.py --config ./configs/inference_pix2pixatt.yaml
python F:\yang_Projects\SynthRad_GAN\train_2d.py --config ./configs/inference_pix2pix.yaml
:: Deactivate the virtual environment
deactivate