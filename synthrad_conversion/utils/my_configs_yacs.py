import datetime
import os
from yacs.config import CfgNode as CN

def init_cfg(yaml_file=None):
    cfg = CN()
    cfg.model_name='' # Unet, AttentionUnet, monai_pix2pix pix2pix WGAN DCGAN; Unet3D
    cfg.server='new' # helix, new, old, home
    cfg.name_prefix='' # for saved name
    cfg.GPU_ID = [0,1]
    cfg.ckpt_path = None
    cfg.mode='train'
    cfg.input_dim = 1
    cfg.real_dim = 1
    cfg.save_dpi=500
    cfg.annotations=''

    # ------------------ #
    # options for data
    # ------------------ #
    cfg.dataset = CN()
    cfg.dataset.data_dir = "" # for spadeldm must be a csv file, otherwise a folder
    cfg.dataset.train_csv = ""
    cfg.dataset.test_csv = ""
    cfg.dataset.ct_dir = ""
    cfg.dataset.mri_dir = ""
    cfg.dataset.mri_mode = 't1_vibe_in'
    cfg.dataset.folder = True
    cfg.dataset.normalize = 'zscore' # 'zscore' or 'minmax
    cfg.dataset.pad='minimum'
    cfg.dataset.rotate = False
    cfg.dataset.spaceXY=0.0
    cfg.dataset.batch_size=8
    cfg.dataset.val_batch_size=1
    cfg.dataset.train_number=1
    cfg.dataset.val_number=1

    cfg.dataset.input_dim=2.0 # 2.0 for 2D, 3.0 for 3D, 3.5 for 3D patches
    cfg.dataset.resized_size=(512,512,None)
    cfg.dataset.patch_size=(None,None,1)
    cfg.dataset.zoom=(1.0,1.0,1.0) # first fixed zoom factor
    cfg.dataset.div_size=(1,1,1)
    cfg.dataset.center_crop=0
    cfg.dataset.shapeAug=True
    cfg.dataset.intensityAug=True
    cfg.dataset.augmentationProb=0.5
    cfg.dataset.rand_min_zoom=(0.5,0.5,1.0) # for random zoom, notice: resized_size x zoom >= patch_size
    cfg.dataset.rand_max_zoom=(1.0,1.0,1.0)

    cfg.dataset.source_name=['mr']
    cfg.dataset.target_name=['ct']
    cfg.dataset.mask_name=['mask']
    cfg.dataset.WINDOW_WIDTH=400
    cfg.dataset.WINDOW_LEVEL=50
    cfg.dataset.tissue_min=-500.0
    cfg.dataset.tissue_max=200.0
    cfg.dataset.bone_min=200.0
    cfg.dataset.bone_max=2000.0
    #cfg.dataset.background=-1024
    cfg.dataset.offset=1024
    cfg.dataset.normmin=-1
    cfg.dataset.normmax=1
    cfg.dataset.scalefactor=1000
    cfg.dataset.MRImax=200
    cfg.dataset.indicator_A="source"
    cfg.dataset.indicator_B="target"
    cfg.dataset.num_workers=1
    # cfg.dataset.load_masks=False
    # cfg.dataset.windowing_and_shifting=False
    # cfg.dataset.input_is_mask=False # for mask to CT generation task, sometimes we only have mask images for inference. In this case, set this to True
    #cfg.dataset.use_all_masks=False # for processing the input data to masks, if we want use all organ masks from totalsegmentator, set this to True
    
    # cfg.dataset.window_width=1
    # ------------------ #
    # options for training
    # ------------------ #
    cfg.train = CN()
    cfg.train.val_epoch_interval=1 # every epoch output validation output and 
    cfg.train.save_last=5 # save last 5 ckpts 
    cfg.train.save_ckpt_interval=1 # save model
    cfg.train.sample_interval=10 # every batches output training output 
    cfg.train.update_D_interval=1
    cfg.train.num_epochs=100
    cfg.train.learning_rate=1e-4  #1e-3
    cfg.train.learning_rate_G=1e-4  #1e-3
    cfg.train.learning_rate_D=5e-5  #1e-3
    cfg.train.beta_1=0.5
    cfg.train.beta_2=0.999
    cfg.train.writeTensorboard=False
    cfg.train.earlystopping_patience=10
    cfg.train.earlystopping_delta=0.0001 # 0.01% decline as threshold
    cfg.train.sample_range_lower=0
    cfg.train.sample_range_upper=1000 # define how many slices to sample
    cfg.train.loss='mse'
    # ------------------ #
    # options for G
    # ------------------ #
    cfg.gen = CN()
    cfg.gen.type='resUnet' # 'resUnet' or 'AttentionUnet'
    cfg.gen.num_channels=(8,16,32,64,128) #(2,4,8,16,32) #(8,16,32,64,128) # notice k=(32,32,None) in DivisiblePadd
    cfg.gen.strides= (2, 2, 2, 2) # (2, 2, 2, 2) or (1, 1, 1, 1) 
    cfg.gen.num_res_units=2
    cfg.gen.act = 'PRELU' # ELU, RELU, LEAKYRELU, PRELU, RELU6, SELU, CELU, GELU, SIGMOID, TANH, SOFTMAX, LOGSOFTMAX, SWISH, MEMSWISH, MISH, GEGLU
    cfg.gen.last_act=None # 'SIGMOID' or 'none'
    cfg.gen.kernel_size=3
    cfg.gen.up_kernel_size=3
    cfg.gen.norm='INSTANCE' # 'BATCH' or 'INSTANCE' or 'none'
    cfg.gen.dropout=0.2
    cfg.gen.bias=True
    cfg.gen.output_min=0
    cfg.gen.output_max=1
    cfg.gen.lambda_bone=10
    cfg.gen.lambda_recon=50.0 # only for pix2pix
    cfg.gen.recon_criterion='MSE' # 'BCE' or 'SSIM' or 'MSE' or 'L1' or 'DiceLoss'
    cfg.gen.in_channels_G=1

    # ------------------ #
    # options for RefineNet
    # ------------------ #
    cfg.refineNet = CN()
    cfg.refineNet.type='resUnet' # 'RefineNet'
    cfg.refineNet.num_channels=(8,16,32,64,128) #(2,4,8,16,32) #(8,16,32,64,128) # notice k=(32,32,None) in DivisiblePadd
    cfg.refineNet.strides= (2, 2, 2, 2) # (2, 2, 2, 2) or (1, 1, 1, 1)
    cfg.refineNet.num_res_units=2
    cfg.refineNet.act = 'SIGMOID' # ELU, RELU, LEAKYRELU, PRELU, RELU6, SELU, CELU, GELU, SIGMOID, TANH, SOFTMAX, LOGSOFTMAX, SWISH, MEMSWISH, MISH, GEGLU
    cfg.refineNet.last_act=None
    cfg.refineNet.kernel_size=3
    cfg.refineNet.up_kernel_size=3
    cfg.refineNet.norm='INSTANCE' # 'BATCH' or 'INSTANCE' or 'none'
    cfg.refineNet.dropout=0.2
    cfg.refineNet.bias=True
    cfg.refineNet.refine_criterion='MSE'
    cfg.refineNet.in_channels_G=1
    cfg.refineNet.output_min=0
    cfg.refineNet.output_max=1
    cfg.refineNet.load_refineNet=True

    # ------------------ #
    # options for D
    # ------------------ #
    cfg.disc = CN()
    cfg.disc.type='monaiDisc' # 'monaiDisc' or 'pix2pixDisc'
    cfg.disc.in_shape=(2, 512, 512)
    cfg.disc.channels=(512, 256, 128, 64, 32, 1)
    cfg.disc.strides=(2, 2, 2, 2, 2, 1)
    cfg.disc.act='PRELU' # 'relu' or 'leakyrelu' or 'selu' or 'tanh' or 'sigmoid' or 'elu' or 'none'
    cfg.disc.kernel_size=3
    cfg.disc.num_res_units=2
    cfg.disc.norm='INSTANCE' # 'BATCH' or 'INSTANCE' or 'none'
    cfg.disc.dropout=0.5
    cfg.disc.last_act=None # 'SIGMOID' or 'none'
    cfg.disc.bias=True
    cfg.disc.hidden_channels_D=64
    cfg.disc.adv_criterion='BCEwithlogits'
    cfg.disc.lambda_adv=1

    cfg.ddpm=CN()
    cfg.ddpm.num_train_timesteps = 1000
    cfg.ddpm.num_inference_steps = 1000
    cfg.ddpm.num_channels = (64, 128, 256, 256)
    cfg.ddpm.attention_levels = (False, False, False, True)
    cfg.ddpm.norm_num_groups = 32
    cfg.ddpm.num_res_units = 2
    cfg.ddpm.num_head_channels=32
    cfg.ddpm.noise_type = 'uniform'
    

    cfg.ldm=CN()
    cfg.ldm.num_train_timesteps = 1000
    cfg.ldm.num_inference_steps = 1000
    cfg.ldm.latent_channels = 32
    cfg.ldm.num_channels_ae = (64, 128, 256, 512)
    cfg.ldm.num_channels_diff = (64, 128, 256, 512)
    cfg.ldm.attention_levels_ae = (False, False, False, False)
    cfg.ldm.attention_levels_diff = (False, False, False, True)
    cfg.ldm.norm_num_groups = 16
    cfg.ldm.cross_attention_dim_diff = 1
    cfg.ldm.num_res_blocks_ae = 2
    cfg.ldm.num_res_blocks_diff = 4
    cfg.ldm.n_epochs_autoencoder = 100
    cfg.ldm.n_epochs_diffusion = 100
    cfg.ldm.ckpt_idx = 4 # we save the last 5 checkpoints in training cycle. 4 is the last one
    cfg.ldm.manual_aorta_diss = -1 # aorta_diss is the label. -1 for automatic loading label from data. 0 or 1 for manual setting label
    cfg.ldm.train_new_different_diff = 0 # if 1, train new different diffusion model

    cfg.validation=CN()
    cfg.validation.evaluate_restore_transforms=True
    cfg.validation.visualize_activation=False
    cfg.validation.manual_aorta_diss = -1 # aorta_diss is the label. -1 for automatic loading label from data. 0 or 1 for manual setting label
    cfg.validation.x_lower_limit=-1
    cfg.validation.x_upper_limit=1
    cfg.validation.y_lower_limit=0
    cfg.validation.y_upper_limit=15000
    cfg.validation.dynamic_range=[-1024., 3000.]
    if yaml_file is not None:
        cfg.merge_from_file(yaml_file)
    # ------------------ #
    # pathes for logs
    # ------------------ #
    return cfg

def config_path(model_name):
    # pathes for logs
    now = datetime.datetime.now()
    date_time = now.strftime('%Y%m%d_%H%M')
    file_prefix = f'{date_time}_{model_name}'
    root=f'./logs/{date_time}_{model_name}'
    saved_logs_folder=os.path.join(root, 'saved_logs')
    parameter_file=os.path.join(saved_logs_folder, f'{file_prefix}_parameters.txt')
    train_loss_file = os.path.join(saved_logs_folder, f'batch_loss_log.txt')
    train_metrics_file = os.path.join(saved_logs_folder, f'batch_metrics_log.txt')
    epoch_loss_file = os.path.join(saved_logs_folder, f'epoch_loss_log.txt')
    saved_img_folder = os.path.join(root, 'saved_outputs')
    saved_model_folder = os.path.join(root, 'saved_models')
    log_dir=os.path.join(root, 'saved_tensorboard')
    
    saved_name_train = os.path.join(saved_logs_folder, f'{file_prefix}_train_loader.csv')
    saved_name_val= os.path.join(saved_logs_folder, f'{file_prefix}_val_loader.csv')
    val_log_file = os.path.join(saved_logs_folder, f'{file_prefix}_val_log.txt')
    model_layer_file = os.path.join(saved_logs_folder, f'model_layer.txt')
    saved_inference_folder = os.path.join(root, 'saved_inference')
    
    return {"saved_logs_folder":saved_logs_folder, 
            "parameter_file":parameter_file, 
            "train_loss_file":train_loss_file, 
            "train_metrics_file": train_metrics_file,
            "epoch_loss_file":epoch_loss_file,
            "saved_img_folder":saved_img_folder, 
            "saved_model_folder":saved_model_folder, 
            "tensorboard_log_dir":log_dir, 
            "saved_name_train":saved_name_train, 
            "saved_name_val":saved_name_val,
            "val_log_file":val_log_file,
            "saved_inference_folder":saved_inference_folder,
            "model_layer_file":model_layer_file,
            "root": root}
