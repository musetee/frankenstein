import argparse
from data.load_data import pre_dataset_for_stylegan
import os

# python savepng.py --path 'D:\Projects\data\Task1\pelvis' --number 5 --prefix 256_5 --resolution 256
# python savepng.py --path 'D:\Projects\data\Task1\pelvis' --number 1 --prefix zscore512 --resolution 512 --normalize zscore
# python savepng.py --path 'D:\Projects\data\Task1\pelvis' --number 1 --prefix minmax512 --resolution 512 --normalize minmax
# python savepng.py --path 'D:\Projects\data\Task1\pelvis' --number 1 --prefix none512 --resolution 512 --normalize none
# python savepng.py --path 'F:\yang_Projects\Datasets\Task1\pelvis' --number 2 --prefix minmax512 --resolution 512 --normalize minmax
# python savepng.py --path 'F:\yang_Projects\Datasets\Task1\pelvis' --number 1 --prefix zscore512 --resolution 512 --normalize zscore
# python savepng.py --path 'C:\Users\56991\Datasets\Task1\pelvis' --number 2 --prefix 256 --resolution 256
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="save png images and labels.")
    #parser.add_argument('--config', default='./data/save_configs/sample.yaml')
    parser.add_argument('--path', default=r'D:\Projects\data\Task1\pelvis')
    parser.add_argument('--number',  default=1, help='number of patients to save')
    parser.add_argument('--prefix',  default='test', help='prefix of the saved folder')
    parser.add_argument('--resolution',  default=256, help='resized resolution of the saved images')
    parser.add_argument('--normalize',  default='zscore', help='normalization method of the saved images')
    args = parser.parse_args()

    data_path=args.path
    train_number=int(args.number)
    prefix=args.prefix
    resolution=int(args.resolution)
    resized_size=(resolution,resolution,None)
    normalize=args.normalize
    #opt=init_cfg(args.config)
    #opt.freeze()
    saved_img_folder=f'.\data\saved_png\{prefix}\png_images'
    saved_label_folder=f'.\data\saved_png\{prefix}\png_labels'
    os.makedirs(saved_img_folder,exist_ok=True)
    os.makedirs(saved_label_folder,exist_ok=True)

    train_ds,train_volume_ds=pre_dataset_for_stylegan(data_path,
                                                    normalize,
                                                    train_number,
                                                    1,
                                                    saved_img_folder,
                                                    saved_label_folder,
                                                    resized_size=resized_size,)
