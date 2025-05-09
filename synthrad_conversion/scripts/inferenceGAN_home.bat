@echo off

:: Activate the virtual environment
D:\Projects\Environments\torch\venv\Scripts\activate

python D:\Projects\SynthRad\train_2D.py --config ./configs/home/1219_inference_attunet.yaml
python D:\Projects\SynthRad\train_2D.py --config ./configs/home/1219_inference_pix2pix_lambda200epoch100.yaml
python D:\Projects\SynthRad\train_2D.py --config ./configs/home/1219_inference_pix2pix_lambda200epoch141.yaml
python D:\Projects\SynthRad\train_2D.py --config ./configs/home/1219_inference_refinenet.yaml
:: Deactivate the virtual environment
deactivate