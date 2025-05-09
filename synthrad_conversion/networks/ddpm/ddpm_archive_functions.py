
def slice_evaluation_for_batch(self, inputs, targets, outputs, intermediates, step):
    for batch_idx in range(targets.shape[0]):
        generated_image=outputs[batch_idx]
        #noise_image=self.noise[batch_idx]
        input_image=inputs[batch_idx]
        orig_image=targets[batch_idx]
        if intermediates is not None:
            num_intermediates = len(intermediates)
            for intermediates_step in range(num_intermediates):
                intermediates_step_batch = intermediates[intermediates_step]
                intermediate = intermediates_step_batch[batch_idx].squeeze().detach().cpu()
                save_single_image(intermediate, 
                                filename=os.path.join(self.img_folder, 
                                f"intermediates_{self.epoch}_{step}_{batch_idx}_{intermediates_step}.{self.imgformat}"), 
                                imgformat=self.imgformat, 
                                dpi=self.dpi)
            
        if self.config.dataset.rotate:
            img_assemble = [
                input_image.squeeze().cpu().detach(), #.permute(1,0)
                orig_image.squeeze().cpu().detach(),  #.permute(1,0)
                generated_image.squeeze().cpu().detach(), #.permute(1,0)
                ]
        else:
            img_assemble = [
                input_image.squeeze().permute(1,0).cpu().detach(),
                orig_image.squeeze().permute(1,0).cpu().detach(), 
                generated_image.squeeze().permute(1,0).cpu().detach(),
                ]
        #self.loss_tracker.update(self.mse_loss.item())
        #self.plot_learning_curves()
        metrics=calculate_mask_metrices(img_assemble[-1],  #*self.config.dataset.scalefactor
                                        img_assemble[1],  #*self.config.dataset.scalefactor
                                        val_masks=None, 
                                        log_file_overall=self.paths["train_metrics_file"], # batch_metrics_log.txt
                                        val_step = f"{self.epoch}, {step}", 
                                        printoutput=False)
        ssim=metrics['ssim']
        psnr=metrics['psnr']
        mae=metrics['mae']
        self.loss_tracker.update_metrics(ssim, psnr, mae)
        
        self.plot_metrics()
        printout=(
            "[Epoch %d] [Step %d] [idx %d] [loss: %f] [ssim: %f] [psnr: %f] [mae: %f] \n"
            % (
                self.epoch,
                step,
                batch_idx,
                self.mse_loss.item(),
                ssim,
                psnr,
                mae,                    
            )
        )
        print(printout)
        with open(self.paths["val_log_file"], 'a') as f: # append mode
            f.write(printout)

        save_png_images=True
        if save_png_images:
            titles = ['MRI', 'CT', self.config.model_name]
            arrange_images_assemble(img_assemble, 
                            titles,
                            saved_name=os.path.join(self.img_folder, 
                            f"compare_epoch_{self.epoch}_{step}_{batch_idx}.{self.imgformat}"), 
                            imgformat=self.imgformat, 
                            dpi=self.dpi)
            '''
            arrange_histograms(
                                img_assemble[1].numpy(), 
                                img_assemble[-1].numpy(), 
                                saved_name=os.path.join(self.hist_folder, 
                                f"histograms_epoch_{self.epoch}_{step}_{batch_idx}.{self.imgformat}"),
                                x_lower_limit=self.config.validation.x_lower_limit, 
                                x_upper_limit=self.config.validation.x_upper_limit,
                                y_lower_limit=self.config.validation.y_lower_limit, 
                                y_upper_limit=self.config.validation.y_upper_limit,
                                )
            '''
            arrange_3_histograms(img_assemble[0].numpy(), 
                                img_assemble[1].numpy(), 
                                img_assemble[2].numpy(), 
                                saved_name=os.path.join(self.hist_folder, 
                                f"histograms_epoch_{self.epoch}_{step}_{batch_idx}.{self.imgformat}"),
                                x_lower_limit=self.config.validation.x_lower_limit, 
                                x_upper_limit=self.config.validation.x_upper_limit,
                                y_lower_limit=self.config.validation.y_lower_limit, 
                                y_upper_limit=self.config.validation.y_upper_limit,
                                )
    mean_ssim, mean_psnr, mean_mae = self.loss_tracker.get_mean_metrics()
    printout_conclusion=(
                "[Epoch %d] [conclusion] [mean ssim: %f] [mean psnr: %f] [mean mae: %f] \n"
                % (
                    self.epoch,
                    mean_ssim,
                    mean_psnr,
                    mean_mae,                    
                )
            )
    print(printout_conclusion)
    with open(self.paths["val_log_file"], 'a') as f: # append mode
        f.write(printout_conclusion)

def evaluate2dBatch(self, batch, outputs_batch, step):
    image_v = batch[self.config.dataset.indicator_B].to(self.device)
    seg_orig = batch[self.config.dataset.indicator_A].to(self.device)
    patient_IDs = batch['patient_ID']
    patient_ID = patient_IDs[-1]
    restore_transforms=self.config.validation.evaluate_restore_transforms
    if restore_transforms:
        self.noise = self.noise*1000
        predicted_reverse = reverse_normalize_data(outputs_batch, mode=self.config.dataset.normalize)
        ground_truth_image_reversed = reverse_normalize_data(image_v, mode=self.config.dataset.normalize)

    seg_reversed = seg_orig
    if VERBOSE: 
        print(seg_reversed.shape, ground_truth_image_reversed.shape, predicted_reverse.shape)

    si = SaveImage(output_dir=f'{self.output_path}',
                separate_folder=False,
                output_postfix=f'synthesized_{step}',
                resample=False)
    si_input = SaveImage(output_dir=f'{self.output_path}',
                separate_folder=False,
                output_postfix=f'input_{step}',
                resample=False)
    si_seg = SaveImage(output_dir=f'{self.output_path}',
                separate_folder=False,
                output_postfix=f'seg_{step}',
                resample=False)
    si(predicted_reverse.permute(1, 2, 3, 0)) #, data['original_affine'][0], data['original_affine'][1]
    si_input(ground_truth_image_reversed.permute(1, 2, 3, 0))
    si_seg(seg_reversed.permute(1, 2, 3, 0))

    self.slice_evaluation_for_batch(seg_reversed, ground_truth_image_reversed, predicted_reverse, intermediates=None, step=step)
            
def evaluateSlice(self, image_losses, val_loss, 
                    input_image_slice, orig_image_slice, generated_image_slice, inputs_volume,
                    epoch,step,batch_idx=0,slice_idx=0):
    input_image=input_image_slice
    orig_image=orig_image_slice
    generated_image=generated_image_slice
    restore_transforms=True
    if restore_transforms:
        orig_image = reverse_normalize_data(orig_image, mode=self.config.dataset.normalize)
        generated_image = reverse_normalize_data(generated_image, mode=self.config.dataset.normalize)
    if self.config.dataset.rotate:
        img_assemble = [
            input_image.squeeze().cpu().detach(), #.permute(1,0)
            orig_image.squeeze().cpu().detach(),  #.permute(1,0)
            generated_image.squeeze().cpu().detach(), #.permute(1,0)
            ]
    else:
        img_assemble = [
            input_image.squeeze().permute(1,0).cpu().detach(),
            orig_image.squeeze().permute(1,0).cpu().detach(), 
            generated_image.squeeze().permute(1,0).cpu().detach(),
            ]
    metrics=calculate_mask_metrices(img_assemble[-1],  #*self.config.dataset.scalefactor
                                    img_assemble[1],  #*self.config.dataset.scalefactor
                                    None, 
                                    self.paths["train_metrics_file"], 
                                    f"{epoch}, {step}", 
                                    printoutput=False)
    ssim=metrics['ssim']
    psnr=metrics['psnr']
    mae=metrics['mae']
    self.loss_tracker.update_metrics(ssim, psnr, mae)
    self.plot_metrics()
    printout=(
        "[Epoch %d] [Batch step %d] [batch_idx %d] [slice_idx %d] [loss: %f] [ssim: %f] [psnr: %f] [mae: %f] \n"
        % (
            epoch,
            step,
            batch_idx,
            slice_idx,
            val_loss.item(),
            ssim,
            psnr,
            mae,                    
        )
    )
    print(printout)
    with open(self.paths["val_log_file"], 'a') as f: # append mode
        f.write(printout)


    imgformat = 'jpg'
    dpi = 50
    titles = ['MRI', 'CT', self.config.model_name]
    img_folder=os.path.join(self.paths["saved_img_folder"], "img")
    os.makedirs(img_folder, exist_ok=True)
    hist_folder=os.path.join(self.paths["saved_img_folder"], "hist")
    os.makedirs(hist_folder, exist_ok=True)
    arrange_images_assemble(img_assemble, 
                    titles,
                    saved_name=os.path.join(img_folder, 
                        f"compare_epoch_{epoch}_{step}_{batch_idx}_{slice_idx}.{imgformat}"), 
                    imgformat=imgformat, dpi=dpi)
    arrange_3_histograms(img_assemble[0].numpy(), 
                            img_assemble[1].numpy(), 
                            img_assemble[-1].numpy(), 
                            saved_name=os.path.join(hist_folder, 
                            f"histograms_epoch_{epoch}_{step}_{batch_idx}_{slice_idx}.{imgformat}"),
                            dpi=dpi, 
                            x_lower_limit=self.config.validation.x_lower_limit, 
                            x_upper_limit=self.config.validation.x_upper_limit,
                            )

    generated_image=generated_image.unsqueeze(-1)
    #print("unsqueezed image shape:", generated_image.shape)
    image_losses.append(val_loss)
    return generated_image

def evaluate25dBatch(self, val_loss, input_image, orig_image, generated_image_batch, inputs, targets,epoch,step):
    image_losses=[]
    output_file=os.path.join(self.output_path, f"test")
    for batch_idx in range(targets.shape[0]):
        for slice_idx in range(targets.shape[-1]):
            print(f"evaluate {batch_idx+1}/{targets.shape[0]} inside batch {step}")
            print(f"for 2.5D image, this is the {slice_idx+1}/{targets.shape[-1]} th slice")
            input_image_slice=inputs[batch_idx,:,:,:,slice_idx]
            orig_image_slice=targets[batch_idx,:,:,:,slice_idx]
            generated_image_slice=generated_image_batch[batch_idx,:,:,:,slice_idx] # B,C,W,H,N
            generated_image = self.evaluateSlice(image_losses, val_loss, 
                    input_image_slice, orig_image_slice, generated_image_slice, inputs,
                    epoch,step,batch_idx,slice_idx)
            try:
                image_volume=torch.cat((image_volume,generated_image),-1)#pay attention to the order!
            except:
                image_volume=generated_image
    
            image_volume.meta = inputs[0].meta
    '''
    ["filename_or_obj", "affine", 
    "original_affine", 
    "spatial_shape", 
    "original_channel_dim", 
    "filename_or_obj", 
    "original_affine", 
    "spatial_shape", 
    "original_channel_dim"]
    '''
    image_volume.meta["filename_or_obj"] = "prediction"
        
    from utils.spacing import resample_ct_volume
    image_volume = resample_ct_volume(image_volume.unsqueeze(0), original_spacing=(1.0, 1.0, 1.0), new_spacing=(1.0, 1.0, 2.5))

    mean_ssim, mean_psnr, mean_mae = self.loss_tracker.get_mean_metrics()
    printout_conclusion=(
                "[Epoch %d] [conclusion] [mean ssim: %f] [mean psnr: %f] [mean mae: %f] \n"
                % (
                    epoch,
                    mean_ssim,
                    mean_psnr,
                    mean_mae,                    
                )
            )
    print(printout_conclusion)
    with open(self.paths["val_log_file"], 'a') as f: # append mode
        f.write(printout_conclusion)
    reversed_image = image_volume
    '''reversed_image = reverse_transforms(output_images=image_volume, 
                                        orig_images=inputs, # output reverse should according to the original inputs MRI
                                        transforms=val_transforms)'''
    '''reversed_image = Rotate90(k=1)(image_volume)'''

    
    #reversed_image = reversed_image.unsqueeze(0)
    #print("reversed image shape:", reversed_image.shape)
    #SaveImage(output_dir=output_file, resample=True)(reversed_image.detach().cpu())
    
    # Create a NIfTI image object
    reversed_image = torch.squeeze(reversed_image.detach().cpu()).numpy()
    reversed_image = np.rot90(reversed_image, axes=(0, 1), k=2)
    import nibabel as nib
    nifti_img = nib.Nifti1Image(reversed_image, affine=np.eye(4))  # You can customize the affine transformation matrix if needed
    # Save the NIfTI image to a file
    output_file_name = output_file + '.nii.gz'
    nib.save(nifti_img, output_file_name)

def postprocess(self, epoch,epoch_num_total, step, loss):
    printout=(
                            "[Epoch %d/%d] [Batch %d] [loss: %f] \n"
                            % (
                                epoch,
                                epoch_num_total,
                                step,
                                loss.item(),                  
                            )
                        )
    with open(self.paths["train_loss_file"], 'a') as f: # append mode
        f.write(printout)

def testdata(self, output_for_check=0, save_folder='test_images'):
    from PIL import Image
    for i, data in enumerate(self.train_loader):
        targets=data['target']
        inputs=data['source']
        print(i, ' targets: ',targets.shape)
        print(i, ' inputs: ',inputs.shape)
        os.makedirs(save_folder,exist_ok=True)
        if output_for_check == 1:
            # save images to file
            for j in range(targets.shape[0]):
                img = targets[j,:,:,:]
                img = img.permute(1,2,0).squeeze().cpu().numpy()
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                img.save(f'{save_folder}/'+str(i)+'_'+str(j)+'.png')
        with open(f'{save_folder}/parameter.txt', 'a') as f:
            f.write('targets batch:' + str(targets.shape)+'\n')
            f.write('inputs batch:' + str(inputs.shape)+'\n')
            f.write('\n')

def plotchain(self):
    device = self.device
    model = self.model
    scheduler = self.scheduler
    inferer = self.inferer
    model.eval()
    noise = torch.randn((1, 1, 64, 64))
    noise = noise.to(device)
    scheduler.set_timesteps(num_inference_steps=1000)
    with autocast(enabled=True):
        image, intermediates = inferer.sample(
            input_noise=noise,   input_image=noise,  diffusion_model=model, scheduler=scheduler, save_intermediates=True, intermediate_steps=100
        )

    chain = torch.cat(intermediates, dim=-1)
    num_images = chain.shape[-1]

    plt.style.use("default")
    plt.imshow(chain[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
    plt.tight_layout()
    plt.axis("off")
    plt.show()

def plot_learning_curves(self):
    epoch_gen_loss_list=self.loss_tracker.generator_losses
    plt.style.use("ggplot")
    plt.title("diffusion training loss", fontsize=20)
    plt.plot(epoch_gen_loss_list, label="training loss")
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Batches", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig(f"{self.output_path}\\diffusion_loss.png")
    plt.close()
    #plt.show()

def plot_metrics(self):
        epoch_ssim_list=self.loss_tracker.ssim
        epoch_psnr_list=self.loss_tracker.psnr
        epoch_mae_list=self.loss_tracker.mae
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        
        axes[0].plot(epoch_ssim_list, color="C0", linewidth=2.0, label="SSIM")
        axes[0].set_title("SSIM", fontsize=20)
        axes[0].set_xlabel("Batches", fontsize=16)
        axes[0].set_ylabel("SSIM", fontsize=16)
        axes[0].legend(prop={"size": 14})
        axes[0].tick_params(axis='both', which='major', labelsize=12)
        
        axes[1].plot(epoch_psnr_list, color="C1", linewidth=2.0, label="PSNR")
        axes[1].set_title("PSNR", fontsize=20)
        axes[1].set_xlabel("Batches", fontsize=16)
        axes[1].set_ylabel("PSNR", fontsize=16)
        axes[1].legend(prop={"size": 14})
        axes[1].tick_params(axis='both', which='major', labelsize=12)
        
        axes[2].plot(epoch_mae_list, color="C2", linewidth=2.0, label="MAE")
        axes[2].set_title("MAE", fontsize=20)
        axes[2].set_xlabel("Batches", fontsize=16)
        axes[2].set_ylabel("MAE", fontsize=16)
        axes[2].legend(prop={"size": 14})
        axes[2].tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        metric_plot_folder=os.path.join(self.paths["saved_img_folder"], f"epoch_{self.epoch}")
        os.makedirs(metric_plot_folder, exist_ok=True)
        plt.savefig(f"{metric_plot_folder}/metrics.png")
        plt.close()
        #plt.show()