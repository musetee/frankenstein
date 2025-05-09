"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import sys
from synthrad_conversion.networks.results_eval import evaluate2dBatch
from synthrad_conversion.networks.basefunc import LossTracker
sys.path.append('synthrad_conversion/networks/gan/cyclegan_and_pix2pix')
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

import torch
try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


def run(input_args, input_config, dataset, paths):
    opt = TestOptions().parse(input_args, input_config.model_name)  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    #dataset = create_dataset(opt)  # create a dataset given opt.
    # dataset_mode and other options
    
    opt.num_test = 1000000
    opt.name = input_config.model_name
    opt.model = input_config.model_name
    opt.direction = 'AtoB'
    opt.n_epochs = input_config.train.num_epochs
    opt.gpu_ids = input_config.GPU_ID # 1 means 3-A6000 in newserver [input_config.GPU_ID] if input is int
    opt.input_nc = 1
    opt.output_nc = 1
    opt.netG = 'unet_256'
    opt.netD = 'basic'
    opt.norm = 'instance'
    opt.dataset_mode = 'aligned'
    opt.batch_size = input_config.dataset.batch_size
    opt.display_server = "http://localhost" #'127.0.0.1'
    opt.display_port = 8097
    opt.checkpoints_dir = input_config.ckpt_path
    opt.results_dir = paths["root"]
    opt.eval = True
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    eval_loss_tracker = LossTracker()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        
        device=model.device
        inputs_batch = data[input_config.dataset.indicator_A].to(device)
        targets_batch = data[input_config.dataset.indicator_B].to(device)
        outputs_batch = model.fake_B
        mse_loss = torch.nn.functional.mse_loss(targets_batch.float(), outputs_batch.float())
        patient_ID_batch = data['patient_ID']
        step=i
        output_path=paths["saved_img_folder"]
        epoch=150
        img_folder=os.path.join(paths["saved_img_folder"], f"epoch_{epoch}", "img")
        os.makedirs(img_folder, exist_ok=True)
        hist_folder=os.path.join(paths["saved_img_folder"], f"epoch_{epoch}", "hist")
        os.makedirs(hist_folder, exist_ok=True)
        imgformat="png"
        dpi=300
        
        
        evaluate2dBatch(
                        inputs_batch, 
                        targets_batch, 
                        outputs_batch, 
                        patient_ID_batch,
                        step,
                        input_config.dataset.val_batch_size,
                        input_config.validation.evaluate_restore_transforms,
                        input_config.dataset.normalize, output_path,
                        epoch, img_folder, imgformat, dpi,
                        input_config.dataset.rotate, 
                        paths["train_metrics_file"], 
                        eval_loss_tracker,
                        mse_loss.item(),
                        hist_folder, 
                        x_lower_limit=input_config.validation.x_lower_limit, 
                        x_upper_limit=input_config.validation.x_upper_limit,
                        y_lower_limit=input_config.validation.y_lower_limit, 
                        y_upper_limit=input_config.validation.y_upper_limit,
                        val_log_file=paths["val_log_file"],
                        val_log_conclusion_file=paths["val_log_file"].replace('val_log','val_conclusion_log'),
                        model_name=input_config.model_name,
                        save_nifti_Batch3D=False,
                        save_nifti_Slice2D=True,
                        save_png_images=False,
                    )


        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()  # save the HTML
if __name__ == '__main__':
    run()