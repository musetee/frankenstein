def run(dataset = 'combined_simplified_csv_seg_assigned'):
    from monai.transforms import SaveImage
    import torch
    import os 
    import sys
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  
    
    from dataprocesser.step1_init_data_list import init_dataset
    from synthrad_conversion.utils.my_configs_yacs import init_cfg, config_path

    
    opt=init_cfg()
    
    server = 'newserver'
    if dataset == 'synthrad_mr2ct':
        opt.dataset.data_dir =  r'D:\Projects\data\synthrad\train\Task1\pelvis'
    elif dataset == 'anish_seg':
        opt.dataset.data_dir = r'D:\Projects\SynthRad\synthrad_conversion\healthy_dissec_home.csv'
    elif dataset == 'combined_simplified_csv_seg_assigned' or dataset == 'combined_simplified_csv_seg_without_assigned_loader':
        # opt.dataset.data_dir = r'synthrad_conversion\combined_new.csv'
        opt.dataset.train_csv = f'synthrad_conversion/datacsv/ct_anika_synthetic_{server}.csv'
        opt.dataset.test_csv = f'synthrad_conversion/datacsv/ct_anika_synthetic_{server}.csv'
        opt.model_name='ddpm'
    opt.dataset.batch_size=4
    opt.dataset.val_batch_size=8
    opt.dataset.normalize='none'
    opt.dataset.zoom=(1.0,1.0,1.0)
    opt.dataset.resized_size=(None,None,None)
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
    step=0
    from tqdm import tqdm
    #print('dataset length:', len(train_loader))
    for data in tqdm(val_loader):
        #print("file:", data['A_paths'])
        test_source=data[opt.dataset.indicator_A]
        test_target=data[opt.dataset.indicator_B]
        test_source=data['mask']
        patient_ID_batch=data['patient_ID']
        batch_size = test_source.shape[0]
        meta = test_target.meta
        def print_all_info(data, title):
            print(f'{title} CT shape: ', data.shape)
            print(f'min,max,mean,std of {title} CT',torch.min(data),torch.max(data),torch.mean(data),torch.std(data))
        print_all_info(test_target, 'target')
        print_all_info(test_source, 'source')
        for batch_idx in range(batch_size):
            total_eval_step = batch_size*step+batch_idx
            patient_ID = patient_ID_batch[batch_idx] if len(patient_ID_batch) == batch_size else patient_ID_batch[0]
            si_input = SaveImage(output_dir=my_paths["saved_img_folder"],
                        separate_folder=False,
                        output_postfix=f'{patient_ID}_mask_{total_eval_step}',
                        resample=False)
            si_input(test_source[batch_idx].unsqueeze(-1),meta)
            if len(test_target.shape)==4: 
                #si(test_target.permute(1, 2, 3, 0)) #, data['original_affine'][0], data['original_affine'][1]
                si_input(test_source[batch_idx].unsqueeze(-1),meta)
        step += 1

if __name__ == '__main__':
    run()