
def run(dataset = 'combined_simplified_csv_seg_assigned'):
    from monai.transforms import SaveImage
    import torch
    import os 
    import sys
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  
    
    from dataprocesser.step1_init_data_list import init_dataset
    from synthrad_conversion.utils.my_configs_yacs import init_cfg, config_path

    def print_all_info(data, title):
        print(f'min,max,mean,std of {title} CT',torch.min(data),torch.max(data),torch.mean(data),torch.std(data))
    opt=init_cfg()
    
    server = 'helix'
    if dataset == 'synthrad_mr2ct':
        opt.dataset.data_dir =  r'D:\Projects\data\synthrad\train\Task1\pelvis'
    elif dataset == 'anish_seg':
        opt.dataset.data_dir = r'D:\Projects\SynthRad\synthrad_conversion\healthy_dissec_home.csv'
    elif dataset == 'combined_simplified_csv_seg_assigned' or dataset == 'combined_simplified_csv_seg_without_assigned_loader':
        # opt.dataset.data_dir = r'synthrad_conversion\combined_new.csv'
        opt.dataset.train_csv = f'synthrad_conversion/datacsv/ct_anika_test_{server}.csv'
        opt.dataset.test_csv = f'synthrad_conversion/datacsv/ct_anika_test_{server}.csv'
        opt.model_name='ddpm'
    opt.dataset.batch_size=4
    opt.dataset.val_batch_size=4
    opt.dataset.normalize='scale2000'
    opt.dataset.zoom=(1.0,1.0,1.0)
    opt.dataset.resized_size=(512,512, None)
    opt.dataset.div_size=(None,None,None)
    opt.dataset.WINDOW_WIDTH=2000
    opt.dataset.WINDOW_LEVEL=0
    opt.dataset.rotate=False

    loader, opt, my_paths = init_dataset(opt=opt, model_name_path = 'testdata', dataset = dataset)
    train_loader=loader.train_loader
    val_loader=loader.val_loader
    train_transforms=loader.train_transforms
    val_transforms=loader.val_transforms
    print("=========================================")
    print("===============testdata==================")
    print("=========================================")
    save_nifti_batch_number=0
    
    step=0
    from tqdm import tqdm
    #print('dataset length:', len(train_loader))
    for data in tqdm(train_loader):
        step += 1
        print("batch: ", step)
        #print("file:", data['A_paths'])
        test_source=data[opt.dataset.indicator_A]
        test_target=data[opt.dataset.indicator_B]
        patient_ID=data['patient_ID']
        target_data=test_target #*4000-1024
        
        if None not in opt.dataset.patch_size:
            patch_size=opt.dataset.patch_size
            H_patch=patch_size[0]
            W_patch=patch_size[1]
            D_patch=patch_size[2]
            
            # check if the input shape is as expected
            if target_data.shape == torch.Size([opt.dataset.batch_size,1,H_patch,W_patch,D_patch]):
                print('shape of ct',target_data.shape)
                print_all_info(test_target, 'target')
                print_all_info(test_source, 'source')
            else:
                raise ValueError("input shape not equal to patch size")
        else:
            print('shape of ct',target_data.shape)
            print_all_info(test_target, 'target')
            print_all_info(test_source, 'source')
            #test_source=monailoader.separate_maps(test_target)
            #test_source=fill_holes_and_extract_contour(test_source)
        
        mask_data = test_source

        printinfo=False
        if printinfo:
            # source image information

            #print(f"source image shape: {mr_data.shape}")
            #print(f"source image affine:\n{mr_data.meta['affine']}")
            #print(f"source image pixdim:\n{mr_data.pixdim}")

            # target image information
            print(f"target image shape: {target_data.shape}")
            print(f"target image affine:\n{target_data.meta['affine']}")
            print(f"target image pixdim:\n{target_data.pixdim}")


            print(f"mask image shape: {mask_data.shape}")
            print(f"mask image affine:\n{mask_data.meta['affine']}")
            print(f"mask image pixdim:\n{mask_data.pixdim}")
        if step<=save_nifti_batch_number : # % 1 == 0
            si = SaveImage(output_dir=my_paths["saved_img_folder"],
                                separate_folder=False,
                                output_postfix=f'target_{step}',
                                resample=False)
            si_input = SaveImage(output_dir=my_paths["saved_img_folder"],
                        separate_folder=False,
                        output_postfix=f'input_{step}',
                        resample=False)
            if len(test_target.shape)==4: 
                si(test_target.permute(1, 2, 3, 0)) #, data['original_affine'][0], data['original_affine'][1]
                si_input(test_source.permute(1, 2, 3, 0))
                save = False
                if save:
                    OutputSourceTargeMaskImages(
                                test_source, 
                                test_target,
                                mask_data,
                                my_paths["saved_img_folder"],
                                patient_ID,
                                step,
                                opt.validation.x_lower_limit,
                                opt.validation.x_upper_limit,
                                opt.validation.y_lower_limit,
                                opt.validation.y_upper_limit)
            elif len(test_target.shape)==5:
                si(test_target.squeeze(0)) #, data['original_affine'][0], data['original_affine'][1]
                si_input(test_source.squeeze(0))

                si_contour = SaveImage(output_dir=my_paths["saved_img_folder"],
                        separate_folder=False,
                        output_postfix=f'contour_{step}',
                        resample=False)
                test_contour = data['mask']
                si_contour(test_contour.squeeze(0))
        else:
            pass
        
def OutputSourceTargeMaskImages(
        test_source, 
        test_target,
        mask_data,
        saved_img_folder,
        patient_ID,
        step,
        x_lower_limit,
        x_upper_limit,
        y_lower_limit,
        y_upper_limit
        ):
    from synthrad_conversion.networks.ddpm.ddpm_mri2ct import arrange_images_assemble
    from synthrad_conversion.utils.evaluate import arrange_1_histogram
    import os
    for i in range(test_target.shape[0]):
        target_i=test_target[i, 0, :, :].permute(1,0)
        source_i=test_source[i, 0, :, :].permute(1,0)
        mask_i=mask_data[i, 0, :, :].permute(1,0)
        img_assemble=[source_i,target_i,mask_i]
        titles=["source","target","Mask" ]
        dpi = 100
        arrange_images_assemble(img_assemble, 
                            titles,
                            saved_name=os.path.join(saved_img_folder,f"{patient_ID[i]}_{step}_{i}.jpg"), 
                            imgformat='jpg', dpi=dpi)
        arrange_1_histogram(target_i, 
                            saved_name=os.path.join(saved_img_folder,f"{patient_ID[i]}_{step}_{i}_hist.jpg"),
                            dpi=dpi, 
                            x_lower_limit=x_lower_limit, 
                            x_upper_limit=x_upper_limit,
                            y_lower_limit=y_lower_limit, 
                            y_upper_limit=y_upper_limit,
                            )

def run3d():
    print('Start testing data...')
    from dataprocesser.list_dataset_synthrad import monai_loader
    from dataprocesser.list_dataset_Anish import anish_loader
    from SynthRad_GAN.dataprocesser.archive.list_dataset_Anish_seg import anish_seg_loader

    from monai.transforms import SaveImage, SobelGradients
    import monai
    monailoader = anish_seg_loader(opt,my_paths,dimension=opt.dataset.input_dim)
    train_loader=monailoader.train_loader
    gradient_calc = SobelGradients(kernel_size=3, spatial_axes=None,)

    
    step=0
    
    si_combined = SaveImage(output_dir=my_paths["saved_img_folder"],
            separate_folder=False,
            output_postfix=f'img_combined',
            resample=False)
    
    filename_list = []
    new_initialization = True
    from tqdm import tqdm
    for data in train_loader:
        #print(data)
        step += 1
        si_input = SaveImage(output_dir=my_paths["saved_img_folder"],
            separate_folder=False,
            output_postfix=f'img_{step}',
            resample=False)
        si_seg = SaveImage(output_dir=my_paths["saved_img_folder"],
                    separate_folder=False,
                    output_postfix=f'seg_{step}',
                    resample=False)
        si_grad = SaveImage(output_dir=my_paths["saved_img_folder"],
                    separate_folder=False,
                    output_postfix=f'grad_{step}',
                    resample=False)
        
        if step == 1:
            print('data:',data)
        #print(data)
        A_data=data['seg']
        B_data=data['img']
        
        # if filename_or_obj changes, then it is a new volume
        #filename_or_obj = B_data.meta['filename_or_obj']
        # filename_or_obj not in filename_list:
        #    filename_list.append(filename_or_obj)
        if new_initialization:
            print('new initialization for the reconstructed volume')
            collected_patches, collected_coords, reconstructed_volume, count_volume = initialize_collection(data)

        patch_coords = data['patch_coords']
        start_pos = data['start_pos']

        collected_patches.append(B_data)
        collected_coords.append(patch_coords)
        Aorta_diss = data['Aorta_diss'].to(torch.float32)
        Aorta_diss=Aorta_diss.unsqueeze(-1)
        
        VERBOSE = True
        save_patch = False
        if VERBOSE:
            print_data_info(A_data)
            print_data_info(B_data)

            print('patch_coords shape:',patch_coords.shape)

            print('Aorta_diss:',Aorta_diss)
            print('Aorta_diss dtype:', Aorta_diss.dtype)
            print(Aorta_diss.shape)
        if save_patch:
            batch_size = A_data.shape[0]
            for batch_idx in range(batch_size):
                si_input(B_data[batch_idx])
                si_seg(A_data[batch_idx])
                grad=gradient_calc(B_data[batch_idx])
                si_grad(grad)

        
        reconstructed_volume_img, count_volume = reconstruct_volume(collected_patches, collected_coords, reconstructed_volume, count_volume)
        reconstructed_volume_img = reconstructed_volume_img.unsqueeze(0)
        # if reconstruction finished for the current volume then save the volume and start a new one
        finished_criteria = torch.all(count_volume == 1)
        #print('finished_criteria:', finished_criteria)
        if finished_criteria:
            new_initialization = True
            si_combined(reconstructed_volume_img, data['img'].meta)
        else:
            new_initialization = False
    '''
    'patch_coords': tensor([[[  0,   1],
        [256, 384],
        [256, 384],
        [  0, 128]],

    [[  0,   1],
        [256, 384],
        [256, 384],
        [128, 256]]],
    shape: torch.Size([2, 4, 2])
        'original_spatial_shape': [tensor([384, 384]), tensor([320, 320]), tensor([256, 256])], 'start_pos': [tensor([0, 0]),

        shape of A torch.Size([2, 1, 128, 128, 128])
    '''

if __name__ == '__main__':
    run()