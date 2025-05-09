    def evaluate2dBatch_old(self,batch,outputs_batch,step):
        image_losses=[]
        targets = self.targets_batch
        inputs = self.inputs_batch
        patient_IDs = batch['patient_ID']
        patient_ID = patient_IDs[-1]
        restore_transforms=self.config.validation.evaluate_restore_transforms

        if restore_transforms:
            self.noise = self.noise*1000

        for batch_idx in range(targets.shape[0]):
            generated_image=outputs_batch[batch_idx]
            noise_image=self.noise[batch_idx]
            input_image=inputs[batch_idx]
            orig_image=targets[batch_idx]
            if self.intermediates is not None:
                num_intermediates = len(self.intermediates)
                for intermediates_step in range(num_intermediates):
                    intermediates_step_batch = self.intermediates[intermediates_step]
                    intermediate = intermediates_step_batch[batch_idx].squeeze().detach().cpu()
                    save_single_image(intermediate, 
                                    filename=os.path.join(self.img_folder, 
                                    f"intermediates_{self.epoch}_{step}_{batch_idx}_{intermediates_step}.{self.imgformat}"), 
                                    imgformat=self.imgformat, 
                                    dpi=self.dpi)
                
            
            if restore_transforms:
                orig_image = reverse_normalize_data(orig_image, mode=self.config.dataset.normalize)
                generated_image = reverse_normalize_data(generated_image, mode=self.config.dataset.normalize)
            if self.config.dataset.rotate:
                img_assemble = [
                    noise_image.squeeze().cpu().detach(), #.permute(1,0)
                    orig_image.squeeze().cpu().detach(),  #.permute(1,0)
                    generated_image.squeeze().cpu().detach(), #.permute(1,0)
                    ]
            else:
                img_assemble = [
                    noise_image.squeeze().permute(1,0).cpu().detach(),
                    orig_image.squeeze().permute(1,0).cpu().detach(), 
                    generated_image.squeeze().permute(1,0).cpu().detach(),
                    ]
            metrics=calculate_mask_metrices(img_assemble[-1],  #*self.config.dataset.scalefactor
                                            img_assemble[1],  #*self.config.dataset.scalefactor
                                            None, 
                                            self.paths["train_metrics_file"], 
                                            f"{self.epoch}, {step}", 
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

            output_file_path=os.path.join(self.paths["saved_img_folder"], f"epoch_{self.epoch}")
            os.makedirs(output_file_path, exist_ok=True)

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
                

            generated_image=generated_image.unsqueeze(-1) # expand the dimension so that the size is as H,W,D
            image_losses.append(self.mse_loss)

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
        
        print("stacked image shape:", image_volume.shape)

        #from utils.spacing import resample_ct_volume
        #image_volume = resample_ct_volume(image_volume.unsqueeze(0), original_spacing=(1.0, 1.0, 1.0), new_spacing=(1.0, 1.0, 2.5))
        #print("resampled image shape:", image_volume.shape)

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
        reversed_image = image_volume
        

        output_file_path=os.path.join(self.paths["saved_img_folder"], f"epoch_{self.epoch}")
        save_method = 'monai_transform_SaveImage'
        # Create a NIfTI image object
        if save_method=='nibabel':
            output_file_name = os.path.join(output_file_path, f"test_{step}.nii.gz")
            reversed_image = torch.squeeze(reversed_image.detach().cpu()).numpy()
            reversed_image = np.rot90(reversed_image, axes=(0, 1), k=2)
            import nibabel as nib
            nifti_img = nib.Nifti1Image(reversed_image, affine=np.eye(4))  # You can customize the affine transformation matrix if needed
            # Save the NIfTI image to a file
            
            nib.save(nifti_img, output_file_name)
        elif save_method == 'monai_transform_SaveImage':
            from monai.transforms import SaveImage
            si = SaveImage(output_dir=output_file_path,
                    separate_folder=False,
                    output_postfix=None,
                    resample=False)
            output_file_name = os.path.join(output_file_path, f"{patient_ID}_{step}")
            si(reversed_image.detach().cpu(), filename=output_file_name)
            output_noise_name = os.path.join(output_file_path, f"{patient_ID}_noise_{step}")
            si(self.noise.permute(1, 2, 3, 0).detach().cpu(), filename=output_noise_name)