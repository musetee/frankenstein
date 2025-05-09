import argparse
from utils.my_configs_yacs import init_cfg

parser = argparse.ArgumentParser(description="StyleGAN pytorch implementation.")
parser.add_argument('--config', default='./configs/sample.yaml')
args = parser.parse_args()

opt=init_cfg(args.config)

opt.freeze()
print('batch_size:',opt.dataset.batch_size)
print('train_number:',opt.dataset.train_number)
print('val_number:',opt.dataset.val_number)
print(opt.path.saved_inference_folder)
print(opt.path.train_loss_file)