import sys
sys.path.append('./networks/ddpm')

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai.utils import first, set_determinism

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from monai.transforms.utils import allow_missing_keys_mode
from torch.nn.parallel import DistributedDataParallel as DDP

from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

from synthrad_conversion.networks.ddpm.transformDiffinferer import transformDiffusionInferer, transformDiffusionInfererWithClass
from synthrad_conversion.networks.results_eval import evaluate2dBatch, evaluate25dBatch,plot_learning_curves
from synthrad_conversion.networks.basefunc import (
    LossTracker,
    EarlyStopping
)
from dataprocesser.step3_build_patch_dataset import patch_2d_from_single_volume, decode_dataset_from_single_volume_batch
import monai
import random
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

if hasattr(torch, 'amp'):
    from torch.amp import autocast
    from torch.amp import GradScaler
    autocast_kwargs = {'device_type': 'cuda', 'enabled': True}
else:
    from torch.cuda.amp import autocast
    from torch.cuda.amp import GradScaler
    autocast_kwargs = {'enabled': True}

VERBOSE = False
VERBOSE_LOSS = True
import platform

# 自动判断操作系统
if platform.system().lower() == "linux":
    USE_TORCH_COMPILE = True
    print("✅ Linux detected: enabling torch.compile()")
else:
    USE_TORCH_COMPILE = False
    print("⚠️ Non-Linux OS detected (likely Windows): disabling torch.compile()")

from synthrad_conversion.networks.model_registry import register_model
@register_model('ddpm2d_seg2med')
class DDPM2DSeg2MedRunner:
    def __init__(self, opt, paths, train_loader, val_loader, train_patient_IDs, test_patient_IDs):
        self.model = DiffusionModel(
            opt, paths,
            train_loader=train_loader,
            val_loader=val_loader,
            train_patient_IDs=train_patient_IDs,
            test_patient_IDs=test_patient_IDs
        )
        self.opt = opt

    def train(self):
        self.model.train()

    def test(self):
        self.model.val()

@register_model('ddpm2d_seg2med_multimodal')
class DDPM2DSeg2Med_multimodal_Runner:
    def __init__(self, opt, paths, train_loader, val_loader, train_patient_IDs, test_patient_IDs):
        self.model = DiffusionModel_multimodal(
            opt, paths,
            train_loader=train_loader,
            val_loader=val_loader,
            train_patient_IDs=train_patient_IDs,
            test_patient_IDs=test_patient_IDs
        )
        self.opt = opt

    def train(self):
        self.model.train()

    def test(self):
        self.model.val()

    def analyse(self):
        self.model.collect_and_visualize_noise_distribution(num_modalities=6, samples_per_modality=50, save_path="analyse_tsne_noise.png")


import torch
import torch.nn.functional as F

def nt_xent_loss(z, labels, temperature=0.5):
    """
    Compute NT-Xent (Normalized Temperature-scaled Cross Entropy Loss)
    Args:
        z: Tensor of shape (B, D), normalized embeddings
        labels: Tensor of shape (B,), integer modality labels
        temperature: Scaling temperature
    Returns:
        scalar contrastive loss
    """
    B = z.shape[0]
    z = F.normalize(z, dim=1)
    similarity = torch.matmul(z, z.T) / temperature  # (B, B)

    # Remove diagonal similarity (self-similarity)
    mask = torch.eye(B, dtype=torch.bool, device=z.device)
    similarity.masked_fill_(mask, float('-inf'))

    # For each i, positive indices are those with the same label
    label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
    label_mask = label_mask & (~mask)  # remove self
    
    # Log-Softmax over similarity matrix rows
    log_prob = F.log_softmax(similarity, dim=1)

    # For each anchor i, select log-probs of positives
    positive_log_probs = log_prob[label_mask]

    # Negative log-likelihood averaged over positives
    loss = -positive_log_probs.mean()
    if VERBOSE_LOSS:
        print('nt-xent label mask:', label_mask)
    return loss



def volume_group_generator(train_loader, volume_batch_size):
    """
    每次从 train_loader 取 volume_batch_size 个 volume，组成一个 group。
    避免一次性 list(train_loader)，节省内存。
    """
    group = []
    for volume in train_loader:
        group.append(volume)
        print(volume['A_paths'])
        if len(group) == volume_batch_size:
            yield group
            group = []
    if group:
        yield group  # 最后一组不足 volume_batch_size 也返回

def checkdata(loader,inputtransforms,output_for_check=1,save_folder='./logs/test_images'):
    from PIL import Image
    import matplotlib
    matplotlib.use('Qt5Agg')
    for i, batch in enumerate(loader):
        images = batch["target"]
        labels = batch["target"]
        
        images=images[:,:,:,:,None]
        try:
            volume=torch.cat((volume,images),-1)
        except:
            volume=images

    volume = volume[0,:,:,:,:] #(B,C,H,W,D)    
    # the input into reverse transform should be in form: 20 is the cropped depth
    # (1, 512, 512, 20) -> (1, 452, 315, 5) C,H,W,D
    print (volume.shape)
    val_output_dict = {"target": volume}
    with allow_missing_keys_mode(inputtransforms):
        reversed_images_dict=inputtransforms.inverse(val_output_dict)

    for i in range(images.shape[0]):
        print(images.shape)
        imgformat='png'
        dpi=300
        os.makedirs(save_folder,exist_ok=True)
        if output_for_check == 1:
            # save images to file
            for j in range(images.shape[-1]):
                saved_name=os.path.join(save_folder,f"{i}_{j}.{imgformat}")
                img = images[:,:,:,j]
                #img =img.squeeze().cpu().numpy()
                img = img.permute(1,2,0).squeeze().cpu().numpy()
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                #img.save(saved_name)

                fig_ct = plt.figure()
                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                            hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.imshow(img, cmap='gray') #.squeeze()
                plt.savefig(saved_name.replace(f'.{imgformat}',f'_reversed.{imgformat}'), format=f'{imgformat}'
                            , bbox_inches='tight', pad_inches=0, dpi=dpi)
                plt.close(fig_ct)

import torch.nn as nn
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
def load_pretrained_model(model, opt, pretrained_path=None):
    if pretrained_path is not None and os.path.exists(pretrained_path):
        latest_ckpt=pretrained_path
        loaded_state = torch.load(latest_ckpt)
        print(f'use pretrained model: {latest_ckpt}') 
        if 'epoch' in loaded_state:
            init_epoch=loaded_state["epoch"] # load or manually set
            print(f'continue from epoch {init_epoch}') 
            #init_epoch = int(input('Enter epoch number: '))
        else:
            print('no epoch information in the checkpoint file')
            init_epoch = int(input('Enter epoch number: '))
        
        if 'global_step' in loaded_state:
            init_step=loaded_state["global_step"] # load or manually set
            print(f'continue from step {init_step}') 
        else:
            print('no epoch information in the checkpoint file')
            init_step = int(input('Enter epoch number: '))

        model.load_state_dict(loaded_state["model"]) #
        opt.load_state_dict(loaded_state["opt"])
    else:
        init_epoch=0
        init_step=0
        #model = model.apply(weights_init)
        print(f'start new training') 

    return model, opt, init_epoch, init_step
def arrange_images_assemble(img_assemble,
                   titles,
                   saved_name,
                   imgformat='jpg',
                   dpi = 500,
                   figsize = (16, 5),
                   ):
        image_number=len(img_assemble)
        fig, axs = plt.subplots(1, image_number, figsize=figsize) # 
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0.1)
        plt.margins(0,0)
        axs = axs.flatten()
        for i in range(image_number):
            axs[i].imshow(img_assemble[i], cmap='gray')
            axs[i].set_title(titles[i])
            axs[i].axis('off')
        # save image as png
        fig.savefig(saved_name, format=f'{imgformat}', bbox_inches='tight', pad_inches=0, dpi=dpi)
        #plt.show()
        plt.close(fig)

def generate_noise(noise_type, targets, device):
    if noise_type == 'uniform':
        noise = torch.rand_like(targets).to(device) 
    elif noise_type == 'normal':
        noise = torch.randn_like(targets).to(device)
    elif noise_type == 'normal_shift_1':
        noise = torch.randn_like(targets)+1
        noise = noise.to(device)
    return noise

class DiffusionModel: #(nn.Module)
    def __init__(self,config,paths,train_loader,val_loader, train_patient_IDs, test_patient_IDs):
        self.config=config
        self.is_ddp = self.configs.is_ddp
        self.rank = self.configs.rank
        self.world_size = self.configs.world_size

        self.paths=paths
        self.output_path=paths["saved_img_folder"] #f'./logs/{args.run_name}/results'
        self.log_path=paths["saved_logs_folder"]
        self.saved_models_name=paths["saved_model_folder"] #f'./logs/{args.run_name}/models'
        self.saved_runs_name=paths["tensorboard_log_dir"] #f'./logs/{args.run_name}/runs'
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_patient_IDs=train_patient_IDs
        self.test_patient_IDs=test_patient_IDs
        self.keys = [self.config.dataset.indicator_A, self.config.dataset.indicator_B]
    
    def init_diffusion_model(self):
        patch_depth = self.config.dataset.patch_size[-1]
        spatial_dims = 2 if patch_depth==1 else 3
        self.model_raw = DiffusionModelUNet(
            spatial_dims=spatial_dims,
            in_channels=2,
            out_channels=1,
            num_res_blocks= self.config.ddpm.num_res_units, # 2
            num_channels= self.config.ddpm.num_channels,
            attention_levels= self.config.ddpm.attention_levels, # (False, False, True, True)
            norm_num_groups=self.config.ddpm.norm_num_groups, # 32
            num_head_channels=self.config.ddpm.num_head_channels, # 8
            with_conditioning=True,
            cross_attention_dim = 1,
        )
        self.inferer = transformDiffusionInferer(self.scheduler)
        self.model = self.model_raw  # 初始赋值为未编译模型

        

    def init_diffusion(self):
        print(f'using {self.config.train.loss} loss for training')
        self.num_train_timesteps=self.config.ddpm.num_train_timesteps
        self.num_inference_steps=self.config.ddpm.num_inference_steps
        self.lr=self.config.train.learning_rate
        self.scheduler = DDPMScheduler(num_train_timesteps=self.num_train_timesteps)
        
        self.init_diffusion_model()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr) 
        
        self.init_gpu_devices()

        
    def init_model_ckpt(self):
        # need to detach model and load state without .module from DDP
        def maybe_strip_module_prefix(state_dict):
            """Automatically remove 'module.' prefix if present on all keys."""
            if all(k.startswith("module.") for k in state_dict.keys()):
                print("model saved in ddp mode, prefixes module. should be removed")
                return {k[len("module."):]: v for k, v in state_dict.items()}
            elif all(k.startswith("_orig_mod.") for k in state_dict.keys()):
                print("model saved in ddp mode, prefixes _orig_mod. should be removed")
                return {k[len("_orig_mod."):]: v for k, v in state_dict.items()}
            return state_dict
        
        def strip_all_prefixes(state_dict, prefixes=("module.", "_orig_mod.")):
            """Recursively strip known prefixes from all keys in the state_dict."""
            new_state_dict = state_dict
            for prefix in prefixes:
                if all(k.startswith(prefix) for k in new_state_dict.keys()):
                    print(f"model saved in ddp mode, prefixes {prefix} should be removed")
                    new_state_dict = {k[len(prefix):]: v for k, v in new_state_dict.items()}
            return new_state_dict

        if self.config.ckpt_path and os.path.exists(self.config.ckpt_path):
            loaded_state = torch.load(self.config.ckpt_path, map_location=self.device)
            print(f"Using pretrained model: {self.config.ckpt_path}")

            if 'epoch' in loaded_state:
                self.init_epoch=loaded_state["epoch"] # load or manually set
                print(f'continue from epoch {self.init_epoch}') 
            else:
                print('no epoch information in the checkpoint file')
                self.init_epoch = int(input('Enter epoch number: '))
            
            if 'global_step' in loaded_state:
                self.init_step=loaded_state["global_step"] # load or manually set
                print(f'continue from step {self.init_step}') 
            else:
                print('no step information in the checkpoint file')
                self.init_step = int(input('Enter step number: '))

            #model_state_dict = maybe_strip_module_prefix(loaded_state["model"])
            #model_state_dict = maybe_strip_module_prefix(model_state_dict)
            model_state_dict = strip_all_prefixes(loaded_state["model"])
            self.model.load_state_dict(model_state_dict)
            self.optimizer.load_state_dict(loaded_state["opt"])
        else:
            self.init_epoch = 0
            self.init_step = 0
            print("No valid checkpoint found, starting fresh.")
    
    def init_gpu_devices(self):
        if self.is_ddp:
            torch.cuda.set_device(self.rank)
            self.device = torch.device(f"cuda:{self.rank}")
            self.model = self.model.to(self.device)
            self.init_model_ckpt()
            if USE_TORCH_COMPILE:
                self.model = torch.compile(self.model)  # ✅ 编译发生在 DDP 前
            
            self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)
        else:
            self.device = torch.device(f'cuda:{self.config.GPU_ID[0]}' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.init_model_ckpt()
            if USE_TORCH_COMPILE:
                self.model = torch.compile(self.model)  
    
    def init_classifier(self, embedding_dim=256*256, num_modalities=6, lr=1e-4):
        """
        Initialize the classifier, move it to GPU, wrap with DDP if needed,
        and initialize its optimizer.

        Args:
            embedding_dim (int): Input dimension (flattened noise_pred).
            num_modalities (int): Number of modality classes.
            lr (float): Learning rate for classifier optimizer.
        """
        # ---- Define classifier head ----
        class ModalityClassifier(nn.Module):
            def __init__(self, in_dim, num_classes):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 256),
                    nn.ReLU(),
                    nn.Linear(256, num_classes)
                )

            def forward(self, x):
                return self.net(x)

        # Instantiate
        self.classifier = ModalityClassifier(embedding_dim, num_modalities)

        # Move to device
        if self.is_ddp:
            torch.cuda.set_device(self.rank)
            self.classifier = self.classifier.to(self.device)
            self.classifier = DDP(self.classifier, device_ids=[self.rank], find_unused_parameters=True)
        else:
            self.classifier = self.classifier.to(self.device)

        # Create optimizer
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)

        print(f"[INFO] Classifier initialized on {self.device}, DDP: {self.is_ddp}")

    def prepare_model_input(self, source):
        return source
    
    def model_inferer_predict(self, targets, sources, model, noise, timesteps, condition):
        return self.inferer(inputs=targets, append_image=sources, diffusion_model=model, noise=noise, timesteps=timesteps, condition=condition)
    
    def model_inferer_sample(self, Aorta_diss):
        if self.save_intermediates:
            image, intermediates = self.inferer.sample(input_noise=self.noise,          
                                    input_image=self.inputs_batch_prepared, 
                                    diffusion_model=self.model, 
                                    scheduler=self.scheduler,
                                    save_intermediates=self.save_intermediates,
                                    intermediate_steps=self.intermediate_steps,
                                    conditioning=Aorta_diss)
        else:
            image = self.inferer.sample(input_noise=self.noise, 
                                    input_image=self.inputs_batch_prepared, 
                                    diffusion_model=self.model, 
                                    scheduler=self.scheduler,
                                    save_intermediates=self.save_intermediates,
                                    intermediate_steps=self.intermediate_steps,
                                    conditioning=Aorta_diss)
            intermediates = None
        return image, intermediates
    
    def _get_first_information(self, debatched_volume):
        inputs_first = debatched_volume[self.config.dataset.indicator_A]
        targets_first = debatched_volume[self.config.dataset.indicator_B]
        patient_ID_first = debatched_volume['patient_ID']
        ad_first = debatched_volume['Aorta_diss']
        modality_first = debatched_volume['modality']
        print('\n first input volume shape:', inputs_first.shape)
        print('first target volume shape:', targets_first.shape)
        print('first patient ID:', patient_ID_first)
        print('first ad:', ad_first)
        print('first modality:', modality_first)
        print('\n input shape:', inputs_first.shape)
        def print_all_infos(data, title):
            print(f'{title}, min,max,mean,std: ', torch.min(data),torch.max(data),torch.mean(data),torch.std(data))
        print_all_infos(inputs_first, 'sources')
        print_all_infos(targets_first, 'targets')

    def _get_modality_condition(self, batch):
        try:
            condition = batch['Aorta_diss'].to(torch.float32).to(self.device)
        except ValueError as e:
            print(f"no condition available, Error: {e}")
            condition = 0
            condition = condition.to(torch.float32).to(self.device)
        condition=condition.unsqueeze(-1).unsqueeze(-1)
    
    def loss_calc(self, noise_pred, noise, modality=None, mode='mse', lambda_contrastive=0.1, classifier=None):
        """
        Calculate training loss.

        Args:
            noise_pred (torch.Tensor): Predicted noise from the model.
            noise (torch.Tensor): Ground truth noise.
            modality (torch.Tensor): Modality labels for contrastive loss.
            mode (str): One of ['mse', 'mse+contrastive'].
            lambda_contrastive (float): Scaling factor for contrastive loss.

        Returns:
            torch.Tensor: Total loss
        """
        loss_mse = F.mse_loss(noise_pred.float(), noise.float())
        
        if mode == 'mse':
            loss_contrastive = 0


        elif mode == 'ntxent':
            temperature = 0.5
            assert modality is not None, "modality labels required"
            z = noise_pred.view(noise_pred.size(0), -1)
            loss_contrastive = nt_xent_loss(z, modality, temperature=temperature)

        else:
            raise ValueError(f"Unsupported loss mode: {mode}")
        
        if VERBOSE_LOSS:
            print('\n')
            print('modality: ', modality)
            print('mse loss:', loss_mse)
            print('contrastive loss:', loss_contrastive)
        return loss_mse + lambda_contrastive * loss_contrastive, loss_mse, loss_contrastive
    
    def train(self):
        self.init_diffusion()
        height, width = self.config.dataset.resized_size[:2]
        embedding_dim = height * width
        
        self.init_classifier(embedding_dim=embedding_dim, num_modalities=6)

        #model = self.model
        #scheduler = self.scheduler
        #optimizer = self.optimizer
        #init_epoch 
        #init_step
        inferer = self.inferer

        logger = SummaryWriter(self.saved_runs_name)
        writeTensorboard=self.config.train.writeTensorboard

        n_epochs = self.config.train.num_epochs
        val_interval = self.config.train.val_epoch_interval
        pretrained_path = self.config.ckpt_path
        epoch_loss_list = []
        
        # 0.1% threshold for early stopping
        early_stopping = EarlyStopping(patience=self.config.train.earlystopping_patience, min_delta=self.config.train.earlystopping_delta) 
        self.loss_tracker = LossTracker()
        scaler = GradScaler()
        total_start = time.time()
        global_step=self.init_step
        val_step_interval = self.val_step_interval

        for continue_epoch in range(n_epochs):
            epoch = continue_epoch + self.init_epoch + 1
            self.epoch = epoch
            epoch_num_total = n_epochs + self.init_epoch

            self.model.train()

            first_batch = next(iter(self.train_loader))
            first_input = first_batch[self.config.dataset.indicator_A]
            first_target = first_batch[self.config.dataset.indicator_B]
            print("original image shape:", first_target.shape)
            
            epoch_loss = 0
            step = 0

            '''for volume_idx, volume_batch in enumerate(self.train_loader): 
                total_volumes = len(self.train_loader)
                

                debatched_volume = decode_dataset_from_single_volume_batch(volume_batch)
                
                volume_batch_dataset = monai.data.Dataset([debatched_volume])
                train_loader_batch = patch_2d_from_single_volume(self.keys, volume_batch_dataset, self.config.dataset.batch_size, self.config.dataset.num_workers)
                print(f"[{self.rank}] training epoch {epoch}: [{volume_idx+1}/{total_volumes}] Processing volume...")
                '''
            ## shuffle_volume
            
            self.config.dataset.batch_size
            for group_idx, volume_batch_group in enumerate(volume_group_generator(self.train_loader, volume_batch_size=4)):
                print(f"[{self.rank}] Training epoch {epoch}, group {group_idx+1}...")
                slice_list = []
                for volume_batch in volume_batch_group:
                    debatched = decode_dataset_from_single_volume_batch(volume_batch)
                    slice_list.extend(debatched if isinstance(debatched, list) else [debatched])
                
                combined_dataset = monai.data.Dataset(slice_list)
                train_loader_batch = patch_2d_from_single_volume(
                    self.keys, combined_dataset,
                    self.config.dataset.batch_size,
                    self.config.dataset.num_workers
                )

                volume_idx = group_idx

                volume_mean_loss = 0
                volume_mean_loss_mse = 0
                volume_mean_loss_constrative = 0
                patch_count = 0
                for batch in train_loader_batch:
                    step += 1
                    patch_count += 1
                    global_step += 1
                    sources = batch[self.config.dataset.indicator_A].to(self.device) # MRI image / Seg image
                    targets = batch[self.config.dataset.indicator_B].to(self.device) # CT image
                    sources = self.prepare_model_input(sources)
                    modality = batch['modality'].to(self.device)
                    
                    condition = self._get_modality_condition(batch)

                    #if global_step==1 and continue_epoch==0:
                    #    self._get_first_information(debatched_volume)

                    if VERBOSE:
                        print('batch:',step,', input shape:', sources.shape)
                    
                    with autocast(**autocast_kwargs):
                        # Generate random noise
                        noise = generate_noise(self.config.ddpm.noise_type, targets, self.device)
                            
                        timesteps = torch.randint(
                            0, inferer.scheduler.num_train_timesteps, (targets.shape[0],), device=targets.device
                        ).long()

                        # Get model prediction
                        noise_pred = self.model_inferer_predict(targets, sources, self.model, noise, timesteps, condition)

                        # loss = F.mse_loss(noise_pred.float(), noise.float())
                        loss, loss_mse, loss_contrastive = self.loss_calc(
                            noise_pred, noise, 
                            modality=modality, 
                            mode=self.config.train.loss, 
                            lambda_contrastive=0.1, 
                            classifier=self.classifier)
                        
                        volume_mean_loss += loss.item()
                        volume_mean_loss_mse += loss_mse
                        volume_mean_loss_constrative += loss_contrastive

                    self.optimizer.zero_grad(set_to_none=True)
                    if self.config.train.loss == 'mse_crossentropy':
                        self.classifier_optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    if self.config.train.loss == 'mse_crossentropy':
                        scaler.step(self.classifier_optimizer)
                    scaler.update()


                    if writeTensorboard:
                        logger.add_scalar("train_batch_loss", loss.item(), global_step=global_step)
                    
                    if (global_step) % val_step_interval == 0:
                        self.model.eval()
                        self.val()
                        val_epoch_loss = self.mse_loss
                        if writeTensorboard:
                            logger.add_scalar("val_epoch_loss", val_epoch_loss, global_step=epoch)
                    epoch_loss += loss.item()
                
                if patch_count > 0:
                    volume_mean_loss /= patch_count
                    volume_mean_loss_mse /= patch_count
                    volume_mean_loss_constrative /= patch_count
                else:
                    print(f"[Warning] Volume {volume_idx+1} had no valid patches.")
                
                output_loss_to_txt=True
                if output_loss_to_txt:
                    printout=(
                                "[Epoch %d/%d] [Volume %d] [loss: %f] [mse: %f] [constrative: %f]\n"
                                % (
                                    epoch,
                                    epoch_num_total,
                                    volume_idx + 1,
                                    volume_mean_loss,
                                    volume_mean_loss_mse,
                                    volume_mean_loss_constrative        
                                )
                            )
                    with open(self.paths["train_loss_file"], 'a') as f: # append mode
                        f.write(printout)
                    print(printout)
            train_epoch_loss = epoch_loss / (step + 1)
            epoch_loss_list.append(train_epoch_loss)
            learningcurve_output_folder=self.log_path

            #epoch_gen_loss_list=self.loss_tracker.generator_losses
            #self.loss_tracker.update(loss.item())

            plot_learning_curves(epoch_loss_list, learningcurve_output_folder)
            
            if writeTensorboard:
                logger.add_scalar("train_epoch_loss", train_epoch_loss, global_step=epoch)
            
            def save_model(path, epoch, global_step):
                model_to_save = getattr(self, "model_raw", self.model)
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model': model_to_save.state_dict(),
                    'opt': self.optimizer.state_dict()
                }, path)

            # ========== 每个 epoch 末尾调用 ==========
            if epoch % self.config.train.save_ckpt_interval == 0:
                path = os.path.join(self.saved_models_name, f"model_{epoch}.pt")
                save_model(path, epoch, global_step)

            if epoch % val_interval == 0:
                self.model.eval()
                self.val()
                val_epoch_loss = self.mse_loss
                if writeTensorboard:
                    logger.add_scalar("val_epoch_loss", val_epoch_loss, global_step=epoch)

            # Early stopping check
            early_stopping(train_epoch_loss)

            if early_stopping.best_loss_updated:
                best_path = os.path.join(self.saved_models_name, "best.pt")
                save_model(best_path, epoch, global_step)

            if early_stopping.early_stop:
                print("Early stopping triggered.")
                total_time = time.time() - total_start
                stop_path = os.path.join(self.saved_models_name, "model_earlystopp.pt")
                save_model(stop_path, epoch, global_step)
                print(f"Training completed, total time: {total_time:.2f} seconds.")
                break
            
        total_time = time.time() - total_start
        print(f"train completed, total time: {total_time}.")    

    def set_inference_parameters(self, create_folder=True):
        self.manual_aorta_diss = self.config.validation.manual_aorta_diss # default should be -1
        if self.manual_aorta_diss < 0:
            print('aorta dissection not manually set, use the value in csv file')
        self.save_intermediates=False
        self.intermediate_steps=50
        
        if create_folder:    
            self.img_folder=os.path.join(self.paths["saved_img_folder"], f"epoch_{self.epoch}", "img")
            self.npy_folder=os.path.join(self.paths["saved_img_folder"], f"epoch_{self.epoch}", "npy")
            self.hist_folder=os.path.join(self.paths["saved_img_folder"], f"epoch_{self.epoch}", "hist")
        
        
            os.makedirs(self.img_folder, exist_ok=True)
            os.makedirs(self.npy_folder, exist_ok=True)
            os.makedirs(self.hist_folder, exist_ok=True)
        self.imgformat = 'jpg'
        self.dpi = 100

    def val(self):
        pretrained_path = self.config.ckpt_path
        if self.config.mode=='test':
            self.init_diffusion()
            self.epoch = self.init_epoch
            save_nifti_Batch3D=False
            save_nifti_Slice2D=True
            save_png_images=False

        elif self.config.mode=='train':
            print(f'validation step in training epoch {self.epoch}')
            save_nifti_Batch3D=True
            save_nifti_Slice2D=False
            save_png_images=True
        else:
            print('validation only for test or train mode')

        self.set_inference_parameters()
        self.saved_logs_name=self.output_path.replace('results','datalogs')
        self.model.eval()
        self.eval_loss_tracker = LossTracker()

        first_batch = next(iter(self.val_loader))
        first_input = first_batch[self.config.dataset.indicator_A]
        first_target = first_batch[self.config.dataset.indicator_B]
        print("original image shape:", first_target.shape)
        print(f'inference val set from {self.config.train.sample_range_lower} to {self.config.train.sample_range_upper} batch')
        for volume_idx, volume_batch in enumerate(self.val_loader): 
            total_volumes = len(self.val_loader)
            print(f"[{self.rank}] validation: [{volume_idx+1}/{total_volumes}] Processing volume...")
            step = 0
            debatched_volume = decode_dataset_from_single_volume_batch(volume_batch)
            inputs_first = debatched_volume[self.config.dataset.indicator_A]
            targets_first = debatched_volume[self.config.dataset.indicator_B]
            patient_ID_first = debatched_volume['patient_ID']
            ad_first = debatched_volume['Aorta_diss']
            print('\n')
            print('patient ID:', patient_ID_first)
            print('input volume shape:', inputs_first.shape)
            print('target volume shape:', targets_first.shape)
            print('ad:', ad_first)

            volum_batch_dataset = monai.data.Dataset([debatched_volume])
            val_loader_batch = patch_2d_from_single_volume(self.keys, volum_batch_dataset, self.config.dataset.val_batch_size, self.config.dataset.num_workers)
            
            for batch in val_loader_batch:
                batch_lower_limit = self.config.train.sample_range_lower
                batch_upper_limit = self.config.train.sample_range_upper
                if step >= batch_lower_limit and step <= batch_upper_limit: 
                    inputs_batch = batch[self.config.dataset.indicator_A].to(self.device)
                    targets_batch = batch[self.config.dataset.indicator_B].to(self.device)
                    patient_ID_batch = batch['patient_ID']
                    outputs_batch = self._sample(batch = batch) 
                    output_sample_results(inputs_batch, targets_batch, patient_ID_batch, outputs_batch, self.epoch, step, self.img_folder, self.npy_folder)  
                    '''if len(outputs_batch.shape)==4:
                        evaluate2dBatch(
                            inputs_batch, 
                            targets_batch, 
                            outputs_batch, 
                            patient_ID_batch,
                            step, self.config.dataset.val_batch_size,
                            self.config.validation.evaluate_restore_transforms,
                            self.config.dataset.normalize, self.output_path,
                            self.epoch, self.img_folder, self.imgformat, self.dpi,
                            self.config.dataset.rotate, 
                            self.paths["train_metrics_file"], 
                            self.eval_loss_tracker,
                            self.mse_loss.item(),
                            self.hist_folder, 
                            x_lower_limit=self.config.validation.x_lower_limit, 
                            x_upper_limit=self.config.validation.x_upper_limit,
                            y_lower_limit=self.config.validation.y_lower_limit, 
                            y_upper_limit=self.config.validation.y_upper_limit,
                            val_log_file=self.paths["val_log_file"],
                            val_log_conclusion_file=self.paths["val_log_file"].replace('val_log','val_conclusion_log'),
                            model_name=self.config.model_name,
                            dynamic_range = self.config.validation.dynamic_range, 
                            save_nifti_Batch3D=save_nifti_Batch3D,
                            save_nifti_Slice2D=save_nifti_Slice2D,
                            save_png_images=save_png_images,
                        )

                    elif len(outputs_batch.shape)==5:
                        evaluate25dBatch(batch, outputs_batch, step)'''
                # If this patient hasn't been processed yet, add to set and update the progress bar
                '''patient_ID_batch = batch["patient_ID"]
                for patient_ID in patient_ID_batch:
                    if patient_ID not in processed_test_patients:
                        processed_test_patients.append(patient_ID)
                        pbar.set_postfix({'pID': f"{patient_ID}"})
                        pbar.update(1)'''
                step += 1
                #pbar.update(1)

    def _sample(self, 
                batch):
        # Sampling
        # targets: CT image, which is  the target
        # inputs: MRI image, which is to be converted to CT image
        
        self.targets_batch = batch[self.config.dataset.indicator_B].to(self.device) # CT image
        self.inputs_batch = batch[self.config.dataset.indicator_A].to(self.device) # MRI image
        self.inputs_batch_prepared = self.prepare_model_input(self.inputs_batch) # one hot processing if needed
        self.noise = generate_noise(self.config.ddpm.noise_type, self.targets_batch, self.device)
        self.scheduler.set_timesteps(num_inference_steps=self.num_inference_steps)
        if self.manual_aorta_diss >= 0:
            Aorta_diss_refer = batch['Aorta_diss'].to(torch.float32).to(self.device)
            aorta_diss_length = len(Aorta_diss_refer)
            #
            Aorta_diss = torch.full((aorta_diss_length,), self.manual_aorta_diss)
            Aorta_diss = Aorta_diss.to(torch.float32).to(self.device)
            Aorta_diss=Aorta_diss.unsqueeze(-1).unsqueeze(-1)
            print(f"set aorta dissertion as {self.manual_aorta_diss} manually")
        else:
            Aorta_diss = batch['Aorta_diss'].unsqueeze(-1).unsqueeze(-1).to(torch.float32).to(self.device)

        with autocast(**autocast_kwargs):
            outputs_batch, self.intermediates = self.model_inferer_sample(Aorta_diss)
        self.mse_loss = F.mse_loss(outputs_batch, self.targets_batch)
        self.targets_batch  = self.targets_batch.detach().cpu()
        self.inputs_batch = self.inputs_batch.detach().cpu()
        outputs_batch = outputs_batch.detach().cpu()   
        return outputs_batch

class DiffusionModel_multimodal(DiffusionModel): #(nn.Module)
    def __init__(self,config,paths,train_loader,val_loader, train_patient_IDs, test_patient_IDs):
        #super(DiffusionModel, self).__init__()
        self.config=config
        self.is_ddp = self.config.is_ddp
        self.rank = self.config.rank
        self.world_size = self.config.world_size
        
        self.paths=paths
        self.output_path=paths["saved_img_folder"] #f'./logs/{args.run_name}/results'
        self.log_path=paths["saved_logs_folder"]
        self.saved_models_name=paths["saved_model_folder"] #f'./logs/{args.run_name}/models'
        self.saved_runs_name=paths["tensorboard_log_dir"] #f'./logs/{args.run_name}/runs'
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_patient_IDs=train_patient_IDs
        self.test_patient_IDs=test_patient_IDs
        self.keys = [self.config.dataset.indicator_A, self.config.dataset.indicator_B]
        self.val_step_interval = 2000

        # initializing the method for guiding diffusion model with conditioning
        # method 1 class_labels: guide by changing time_embedding temb
        # method 2 context: guide by participate in attention module
        self.guide_by_context = False
        self.with_conditioning = True if self.guide_by_context else False
        self.cross_attention_dim = 1 if self.guide_by_context else None

    def init_diffusion_model(self):
        print('init diffusion model, with guide_by_context is', self.guide_by_context)
        patch_depth = self.config.dataset.patch_size[-1]
        spatial_dims = 2 if patch_depth==1 else 3
        self.model = DiffusionModelUNet(
            spatial_dims=spatial_dims,
            in_channels=2,
            out_channels=1,
            num_res_blocks= self.config.ddpm.num_res_units, # 2
            num_channels= self.config.ddpm.num_channels,
            attention_levels= self.config.ddpm.attention_levels, # (False, False, True, True)
            norm_num_groups=self.config.ddpm.norm_num_groups, # 32
            num_head_channels=self.config.ddpm.num_head_channels, # 8
            with_conditioning=self.with_conditioning,
            cross_attention_dim = self.cross_attention_dim,
        )

        self.inferer = transformDiffusionInfererWithClass(self.scheduler)

    def model_inferer_predict(self, targets, sources, model, noise, timesteps, condition):
        if self.guide_by_context:
            inferer_result = self.inferer(inputs=targets, append_image=sources, diffusion_model=model, noise=noise, timesteps=timesteps, context=condition)
        else:
            inferer_result = self.inferer(inputs=targets, append_image=sources, diffusion_model=model, noise=noise, timesteps=timesteps, class_labels=condition)
        return inferer_result
    
    def model_inferer_sample(self, conditioning, progress_callback=None):
        # 构造公共参数
        kwargs = {
            "input_noise": self.noise,
            "input_image": self.inputs_batch_prepared,
            "diffusion_model": self.model,
            "scheduler": self.scheduler,
            "save_intermediates": self.save_intermediates,
            "intermediate_steps": self.intermediate_steps,
            "progress_callback": progress_callback,
        }

        # 根据模式添加正确的引导类型
        if self.guide_by_context:
            kwargs["context"] = conditioning
        else:
            kwargs["class_labels"] = conditioning

        # 执行推理
        result = self.inferer.sample(**kwargs)

        # 返回统一格式
        if self.save_intermediates:
            image, intermediates = result
        else:
            image, intermediates = result, None

        return image, intermediates

    def _get_modality_condition(self, batch):
        modality = batch['modality'].long().to(self.device)
        modality = modality.unsqueeze(-1).unsqueeze(-1)  # shape [B, 1, 1]
        return modality
    
    def collect_and_visualize_noise_distribution(self, num_modalities=6, samples_per_modality=50, save_path="analyse_tsne_noise.png"):
        self.init_diffusion()
        model = self.model
        inferer= self.inferer
        model.eval()
        noise_embeddings = []
        modality_labels = []
        collected_per_mod = {i: 0 for i in range(num_modalities)}
        device = self.device
        with torch.no_grad():
            for volume_idx, volume_batch in enumerate(self.train_loader): 
                total_volumes = len(self.train_loader)
                print(f"[{self.rank}] analysing: [{volume_idx+1}/{total_volumes}] Processing volume...")
                step = 0
                debatched_volume = decode_dataset_from_single_volume_batch(volume_batch)
                
                volume_batch_dataset = monai.data.Dataset([debatched_volume])
                train_loader_batch = patch_2d_from_single_volume(self.keys, volume_batch_dataset, self.config.dataset.batch_size, self.config.dataset.num_workers)

                for batch in train_loader_batch:
                    inputs = batch[self.config.dataset.indicator_A].to(device)
                    targets = batch[self.config.dataset.indicator_B].to(device)
                    modality = batch['modality'].to(device)
                    condition = self._get_modality_condition(batch)

                    batch_size = inputs.size(0)
                    #timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (batch_size,), device=device)
                    timesteps = torch.full((batch_size,), fill_value=200, dtype=torch.long, device=device)
                    noise = torch.randn_like(targets)
                    noise_pred = self.model_inferer_predict(targets, inputs, self.model, noise, timesteps, condition)

                    flat_pred = noise_pred.view(batch_size, -1).cpu().numpy()
                    modality_np = modality.cpu().numpy()

                    for i in range(batch_size):
                        mod = modality_np[i]
                        if collected_per_mod[mod] < samples_per_modality:
                            noise_embeddings.append(flat_pred[i])
                            modality_labels.append(mod)
                            collected_per_mod[mod] += 1

                    if all(v >= samples_per_modality for v in collected_per_mod.values()):
                        break

        # Do t-SNE
        X = np.array(noise_embeddings)
        y = np.array(modality_labels)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_2d = tsne.fit_transform(X)

        plt.figure(figsize=(8, 6))
        for i in range(num_modalities):
            plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], label=f'Modality {i}', alpha=0.6)
        plt.legend()
        plt.title("Predicted Noise Distribution Across Modalities (t-SNE)")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        # PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        plt.figure(figsize=(8, 6))
        for i in range(num_modalities):
            plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=f'Modality {i}', alpha=0.6)
        plt.legend()
        plt.title("PCA of predicted noise")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.tight_layout()
        pca_path = save_path.replace("tsne", "pca")
        plt.savefig(pca_path)
        plt.close()

    def _sample(self, 
                batch):
        # Sampling
        # targets: CT image, which is  the target
        # inputs: MRI image, which is to be converted to CT image
        
        self.targets_batch = batch[self.config.dataset.indicator_B].to(self.device) # CT image
        self.inputs_batch = batch[self.config.dataset.indicator_A].to(self.device) # MRI image
        self.inputs_batch_prepared = self.prepare_model_input(self.inputs_batch) # one hot processing if needed
        self.noise = generate_noise(self.config.ddpm.noise_type, self.targets_batch, self.device)
        self.scheduler.set_timesteps(num_inference_steps=self.num_inference_steps)
        condition = self._get_modality_condition(batch)

        with autocast(**autocast_kwargs):
            outputs_batch, self.intermediates = self.model_inferer_sample(condition)
        self.mse_loss = F.mse_loss(outputs_batch, self.targets_batch)
        self.targets_batch  = self.targets_batch.detach().cpu()
        self.inputs_batch = self.inputs_batch.detach().cpu()
        outputs_batch = outputs_batch.detach().cpu()   
        return outputs_batch

import numpy as np
import cv2

import numpy as np
import cv2

def resize_with_zoom_and_pad(source, zoom_factor, resized_size):
    # Step 1: 缩放
    h, w = source.shape[:2]
    zoomed_h, zoomed_w = int(h * zoom_factor), int(w * zoom_factor)
    zoomed = cv2.resize(source, (zoomed_w, zoomed_h), interpolation=cv2.INTER_AREA)

    final_h, final_w = resized_size
    final = np.zeros((final_h, final_w), dtype=zoomed.dtype)

    # Step 2: 中心裁剪或填充
    # 计算 zoomed 和 final 的差异
    dh = final_h - zoomed_h
    dw = final_w - zoomed_w

    # 如果需要填充
    pad_top = max(dh // 2, 0)
    pad_left = max(dw // 2, 0)

    # 如果需要裁剪
    crop_top = max(-dh // 2, 0)
    crop_left = max(-dw // 2, 0)

    # 裁剪后 zoomed 的尺寸
    cropped = zoomed[crop_top:crop_top + min(final_h, zoomed_h),
                     crop_left:crop_left + min(final_w, zoomed_w)]

    # 插入到最终图像中
    insert_h, insert_w = cropped.shape[:2]
    final[pad_top:pad_top + insert_h, pad_left:pad_left + insert_w] = cropped

    return final



class DiffusionModel_multimodal_for_app(DiffusionModel_multimodal):
    def inference(self, source, modality, streamlit_callback):
        self.init_diffusion()
        self.set_inference_parameters(create_folder=False)
        self.model.eval()
        
        
        
        source = source.copy()
        source = source.astype(float)
        resized_size = self.config.dataset.resized_size[:2]
        zoom_factor= self.config.dataset.zoom[0] 
        
        
        prior_modality_norm_dict = {
            0: {'min': -300, 'max': 700},   # CT WW=1000, WL=200
            1: {'min': 0, 'max': 9},       # T1
            2: {'min': 0, 'max': 28},       # T2
            3: {'min': 0, 'max': 9},       # VIBE-IN
            4: {'min': 0, 'max': 10},       # VIBE-OPP
            5: {'min': 0, 'max': 6},       # DIXON
        }
        prior_params = prior_modality_norm_dict[modality]
        x_prior = np.clip(source, prior_params['min'], prior_params['max'])
        x_prior = (x_prior - prior_params['min']) / (prior_params['max'] - prior_params['min'])
        
        x_prior = resize_with_zoom_and_pad(x_prior, zoom_factor=zoom_factor, resized_size=resized_size)
        
        print("input range:", np.min(source), np.max(source))
        print("input size:", x_prior.shape)
        
        self.inputs_batch_prepared = torch.from_numpy(np.expand_dims(np.expand_dims(x_prior, axis=0), axis=0)).float().to(self.device)
        self.noise = generate_noise(self.config.ddpm.noise_type, self.inputs_batch_prepared, self.device)
        self.scheduler.set_timesteps(num_inference_steps=self.num_inference_steps)
        modality = torch.tensor([modality], dtype=torch.long).to(self.device)
        print()
        with autocast(**autocast_kwargs):
            outputs_batch, self.intermediates = self.model_inferer_sample(modality, streamlit_callback)
        return outputs_batch

def output_sample_results(inputs_batch, targets_batch, patient_ID_batch, outputs_batch, epoch, step, img_folder, npy_folder):
    import os
    import numpy as np
    import imageio

    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(npy_folder, exist_ok=True)

    for i in range(outputs_batch.shape[0]):
        input_img = inputs_batch[i, 0]
        target_img = targets_batch[i, 0]
        output_img = outputs_batch[i, 0]
        patient_id = patient_ID_batch[i]

        # Determine if 2D or 3D
        is_3d = len(output_img.shape) == 3  # (H, W, D) if 3D, (H, W) if 2D

        # Save npy files
        np.save(os.path.join(npy_folder, f"epoch_{epoch}_{patient_id}_step_{step}_input.npy"), input_img.cpu().numpy())
        np.save(os.path.join(npy_folder, f"epoch_{epoch}_{patient_id}_step_{step}_target.npy"), target_img.cpu().numpy())
        np.save(os.path.join(npy_folder, f"epoch_{epoch}_{patient_id}_step_{step}_output.npy"), output_img.cpu().numpy())

        # Save preview PNGs
        if is_3d:
            mid_slice = output_img.shape[-1] // 2  # axial middle slice
            input_preview = input_img[:, :, mid_slice]
            target_preview = target_img[:, :, mid_slice]
            output_preview = output_img[:, :, mid_slice]
        else:
            input_preview = input_img
            target_preview = target_img
            output_preview = output_img

        # Helper function to save grayscale image
        def save_grayscale_image(tensor, filename):
            img = tensor.squeeze().cpu().numpy()            # shape: (H, W)
            img = np.transpose((img * 255).clip(0, 255).astype(np.uint8)) # 转换为 0-255 范围的 uint8 图像
            imageio.imwrite(filename, img)

        # 保存 input、target、output 图像
        save_grayscale_image(input_preview, os.path.join(img_folder, f"{patient_id}_input.png"))
        save_grayscale_image(target_preview, os.path.join(img_folder, f"{patient_id}_target.png"))
        save_grayscale_image(output_preview, os.path.join(img_folder, f"{patient_id}_output.png"))


