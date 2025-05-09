@echo off

:: Activate the virtual environment
E:/Projects/yang_proj/torch/venv/Scripts/Activate.ps1

python E:\Projects\yang_proj\SynthRad_GAN\train_2D.py --config ./configs/newserver/infer_AttentionUnet.yaml
python E:\Projects\yang_proj\SynthRad_GAN\train_2D.py --config ./configs/newserver/infer_pix2pix_lambda50.yaml
python E:\Projects\yang_proj\SynthRad_GAN\train_2D.py --config ./configs/newserver/infer_pix2pix_lambda50scratch.yaml
python E:\Projects\yang_proj\SynthRad_GAN\train_2D.py --config ./configs/newserver/infer_pix2pix_lambda200.yaml
python E:\Projects\yang_proj\SynthRad_GAN\train_2D.py --config ./configs/newserver/infer_refinenet.yaml
:: Deactivate the virtual environment
deactivate