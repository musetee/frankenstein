from torch.utils.data import DataLoader
import torch 
from tqdm import tqdm
import os
import json
from dataprocesser.step1_init_data_list import init_dataset
VERBOSE=False
num_workers=1

def run():
    from synthrad_conversion.utils.my_configs_yacs import init_cfg, config_path
    opt=init_cfg()
    dataset = 'anish_seg'
    opt.dataset.data_dir = r'D:\Projects\SynthRad\synthrad_conversion\healthy_dissec_home.csv'
    opt.dataset.batch_size=16
    opt.dataset.val_batch_size=16
    opt.dataset.normalize='scale2000'
    opt.dataset.zoom=(0.5,0.5,1.0)
    opt.dataset.resized_size=(256,256, None)
    opt.dataset.div_size=(None,None,None)
    opt.dataset.WINDOW_WIDTH=2000
    opt.dataset.WINDOW_LEVEL=0
    opt.dataset.rotate=False
    loader, opt, my_paths = init_dataset(opt=opt, model_name_path = 'dataset_slicing', dataset = dataset)
    save_slices_nifti_and_csv(save_output_path='./data/try/train', dataset=loader.train_volume_ds, indicator_A=opt.dataset.indicator_A, indicator_B=opt.dataset.indicator_B)

def save_augemented_dataset_nifti(train_loader, save_output_path, case=0):
    from monai.transforms import SaveImage
    step = 0
    with torch.no_grad():
        for data in train_loader:
            si_input = SaveImage(output_dir=f'{save_output_path}',
                separate_folder=False,
                output_postfix=f'', # aug_{step}
                resample=False)
            si_seg = SaveImage(output_dir=f'{save_output_path}',
                separate_folder=False,
                output_postfix=f'', # aug_{step}
                resample=False)
            
            image_batch = data['img'].squeeze()
            seg_batch = data['seg'].squeeze()
            file_path_batch = data['B_paths']
            batch_size = len(file_path_batch)

            for i in range(batch_size):
                step += 1

                file_path = file_path_batch[i]
                image = image_batch[i]
                seg = seg_batch[i]

                patient_ID = os.path.splitext(os.path.basename(file_path))[0]
                save_name_img = patient_ID + '_aug_' + str(case) + '_' + str(step)
                save_name_img = os.path.join(save_output_path, save_name_img)

                save_name_seg = patient_ID + '_aug_' + str(case) + '_' + str(step) + '_seg'
                save_name_seg = os.path.join(save_output_path, save_name_seg)
                
                si_input(image.unsqueeze(0), data['img'].meta, filename=save_name_img)
                si_seg(seg.unsqueeze(0), data['seg'].meta, filename=save_name_seg)

def save_slices_nifti_and_json(save_output_path, dataset, indicator_A='seg', indicator_B='img'):
    from monai.transforms import SaveImage
    loader = DataLoader(
            dataset, #self.train_volume_ds
            num_workers=num_workers, 
            batch_size=1,
            pin_memory=torch.cuda.is_available())
    
    output_json_file = os.path.join(save_output_path, 'dataset.json')
    output_patient_info_file = os.path.join(save_output_path, 'patient_info.txt')
    dataset_list = []
    with torch.no_grad():
        for data in tqdm(loader):
            si = SaveImage(output_dir=f'{save_output_path}',
                separate_folder=False,
                print_log=False,
                output_postfix=f'', 
                resample=False)
            
            seg_batch = data[indicator_A] #.squeeze()
            image_batch = data[indicator_B] #.squeeze()
        
            Aorta_diss_batch = data['Aorta_diss'] #.cpu().detach().numpy()
            patient_ID_batch = data['patient_ID']
            

            b, c, h, w, d= image_batch.size()

            file_path_batch = data['B_paths']
            batch_size = len(file_path_batch)
            
            for i in range(batch_size):
                file_path = file_path_batch[i]
                image = image_batch[i]
                seg = seg_batch[i]
                Aorta_diss = Aorta_diss_batch[i].item()
                patient_ID = patient_ID_batch[i]

                patient_info = (f"patient_ID: {patient_ID}, Aorta_diss: {Aorta_diss} \n")
                with open(output_patient_info_file, 'a') as f:
                    f.write(patient_info)
                    
                if VERBOSE:
                    print('Aorta_diss:', Aorta_diss)
                    print('patient_ID:', patient_ID)

                os.makedirs(os.path.join(save_output_path, patient_ID), exist_ok=True)

                for j in range(d):
                    save_name_img = patient_ID + '_' + str(j)
                    save_name_img = os.path.join(save_output_path, patient_ID, save_name_img)
                    
                    save_name_seg = patient_ID + '_seg_' + str(j)
                    save_name_seg = os.path.join(save_output_path, patient_ID, save_name_seg)
                    
                    image_slice = image[:,:,:,j]
                    seg_slice = seg[:,:,:,j]
                    if VERBOSE:
                        print(f"target image shape: {image_slice.shape}")
                        print(f"target image affine:\n{image_slice.meta['affine']}")
                        print(f"target image pixdim:\n{image_slice.pixdim}")

                    si(image_slice.unsqueeze(-1), image_batch.meta, filename=save_name_img) # 
                    si(seg_slice.unsqueeze(-1), seg_batch.meta, filename=save_name_seg) # 

                    # Create the entry
                    entry = {
                        'ground_truth': save_name_img + '.nii.gz',
                        'observation': save_name_seg + '.nii.gz',
                        'patient_ID': patient_ID,
                        'Aorta_diss': Aorta_diss
                    }

                    dataset_list.append(entry)
                    # Save the dataset list as a JSON file
    with open(output_json_file, 'w') as json_file:
        json.dump(dataset_list, json_file, indent=4)

def save_slices_nifti_and_csv(save_output_path, dataset, indicator_A='seg', indicator_B='img'):
    from monai.transforms import SaveImage
    import csv
    loader = DataLoader(
            dataset, #self.train_volume_ds
            num_workers=num_workers, 
            batch_size=1,
            pin_memory=torch.cuda.is_available())
    
    output_csv_file = os.path.join(save_output_path, 'dataset.csv')
    output_patient_info_file = os.path.join(save_output_path, 'patient_info.csv')
    dataset_list = []
    patient_info_list = []
    with torch.no_grad():
        for data in tqdm(loader):
            si = SaveImage(output_dir=f'{save_output_path}',
                separate_folder=False,
                print_log=False,
                output_postfix=f'', 
                resample=False)
            
            seg_batch = data[indicator_A] #.squeeze()
            image_batch = data[indicator_B] #.squeeze()
        
            Aorta_diss_batch = data['Aorta_diss'] #.cpu().detach().numpy()
            patient_ID_batch = data['patient_ID']
            

            b, c, h, w, d= image_batch.size()

            file_path_batch = data['B_paths']
            batch_size = len(file_path_batch)
            
            for i in range(batch_size):
                file_path = file_path_batch[i]
                image = image_batch[i]
                seg = seg_batch[i]
                Aorta_diss = Aorta_diss_batch[i].item()
                patient_ID = patient_ID_batch[i]

                patient_info = [patient_ID,Aorta_diss]
                if VERBOSE:
                    print('Aorta_diss:', Aorta_diss)
                    print('patient_ID:', patient_ID)
                os.makedirs(os.path.join(save_output_path, patient_ID), exist_ok=True)

                for j in range(d):
                    save_name_img = patient_ID + '_' + str(j)
                    save_name_img = os.path.join(save_output_path, patient_ID, save_name_img)
                    
                    save_name_seg = patient_ID + '_seg_' + str(j)
                    save_name_seg = os.path.join(save_output_path, patient_ID, save_name_seg)
                    
                    image_slice = image[:,:,:,j]
                    seg_slice = seg[:,:,:,j]
                    if VERBOSE:
                        print(f"target image shape: {image_slice.shape}")
                        print(f"target image affine:\n{image_slice.meta['affine']}")
                        print(f"target image pixdim:\n{image_slice.pixdim}")

                    si(image_slice.unsqueeze(-1), image_batch.meta, filename=save_name_img) # 
                    si(seg_slice.unsqueeze(-1), seg_batch.meta, filename=save_name_seg) # 

                    # Create the entry
                    entry = [
                        save_name_img + '.nii.gz', # 'ground_truth': 
                        save_name_seg + '.nii.gz', # 'observation': 
                        patient_ID, # 'patient_ID': 
                        Aorta_diss # 'Aorta_diss': 
                    ]

                    
                    dataset_list.append(entry)
                    # Save the dataset list as a JSON file
                patient_info_list.append(patient_info)

    with open(output_patient_info_file, 'w', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['patient_ID', 'Aorta_diss'])
        csvwriter.writerows(patient_info_list)

    with open(output_csv_file, 'w', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['img', 'seg', 'patient_ID', 'Aorta_diss'])
        csvwriter.writerows(dataset_list)  

def convert_slices_as_dicom_and_json(train_volume_ds, save_output_path):
    from nii2dcm.run import run_nii2dcm
    train_loader = DataLoader(
        train_volume_ds,
        num_workers=num_workers,
        batch_size=1,
        pin_memory=torch.cuda.is_available()
    )
    
    output_json_file = os.path.join(save_output_path, 'dataset.json')
    with torch.no_grad():
        for data in train_loader:
            Aorta_diss_batch = data['Aorta_diss']
            patient_ID_batch = data['patient_ID']
            
            seg_path_batch = data['A_paths']
            file_path_batch = data['B_paths']
            batch_size = len(file_path_batch)
            
            
            for i in range(batch_size):
                seg_path = seg_path_batch[i]
                file_path = file_path_batch[i]
                
                Aorta_diss = Aorta_diss_batch[i].item()
                patient_ID = patient_ID_batch[i]


                print('Aorta_diss:', Aorta_diss)
                print('patient_ID:', patient_ID)

                patient_output_dir = os.path.join(save_output_path, patient_ID)
                os.makedirs(patient_output_dir, exist_ok=True)

                run_nii2dcm(file_path, patient_output_dir, dicom_type=None, ref_dicom_file=None)
                run_nii2dcm(seg_path, patient_output_dir, dicom_type=None, ref_dicom_file=None)
import datetime
def save_dicom(slice_data, file_name, patient_ID, slice_idx, segmentation=False):
    import pydicom
    from pydicom.dataset import Dataset
    """ Function to save a slice as a DICOM file """
    # Create the FileDataset (DICOM file) with basic metadata
    meta = Dataset()
    meta.PatientID = patient_ID
    meta.SliceLocation = slice_idx
    meta.Modality = "CT" if not segmentation else "SEG"  # Set modality appropriately
    meta.ContentDate = datetime.now().strftime('%Y%m%d')
    meta.ContentTime = datetime.now().strftime('%H%M%S')

    # Here, you would set other metadata, such as Study ID, Series Number, etc.
    
    # Convert the pixel data to uint16 format as expected in most DICOM images
    # slice_data = np.clip(slice_data, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    
    # Assign the pixel data and metadata
    meta.Rows, meta.Columns = slice_data.shape
    meta.PixelData = slice_data.tobytes()
    meta.BitsAllocated = 16
    meta.BitsStored = 16
    meta.HighBit = 15
    meta.PixelRepresentation = 1  # 1 for signed integers
    meta.SamplesPerPixel = 1
    meta.PhotometricInterpretation = "MONOCHROME2"

    meta.is_little_endian = True
    meta.is_implicit_VR = True
    # Save the DICOM file
    pydicom.filewriter.dcmwrite(file_name, meta)

def save_slices_as_dicom_and_json(train_volume_ds, save_output_path, indicator_A='seg', indicator_B='img'):
    train_loader = DataLoader(
        train_volume_ds,
        num_workers=num_workers,
        batch_size=1,
        pin_memory=torch.cuda.is_available()
    )
    
    output_json_file = os.path.join(save_output_path, 'dataset.json')
    with torch.no_grad():
        for data in train_loader:
            seg_batch = data[indicator_A]  # Segmentation batch
            image_batch = data[indicator_B]  # Image batch
            Aorta_diss_batch = data['Aorta_diss']
            patient_ID_batch = data['patient_ID']
            
            b, c, h, w, d = image_batch.size()

            file_path_batch = data['B_paths']
            batch_size = len(file_path_batch)
            
            for i in range(batch_size):
                file_path = file_path_batch[i]
                image = image_batch[i]
                seg = seg_batch[i]
                Aorta_diss = Aorta_diss_batch[i].item()
                patient_ID = patient_ID_batch[i]

                print('Aorta_diss:', Aorta_diss)
                print('patient_ID:', patient_ID)

                patient_output_dir = os.path.join(save_output_path, patient_ID)
                os.makedirs(patient_output_dir, exist_ok=True)
                
                patient_seg_output_dir = os.path.join(save_output_path, patient_ID+'_seg')
                os.makedirs(patient_seg_output_dir, exist_ok=True)
                for j in range(d):
                    # Extract the image slice and segmentation slice
                    image_slice = image[:, :, :, j].squeeze().cpu().numpy()  # Convert tensor to numpy
                    seg_slice = seg[:, :, :, j].squeeze().cpu().numpy()

                    # Generate file names
                    dicom_file_name_img = os.path.join(patient_output_dir, f'{patient_ID}_slice_{j}.dcm')
                    dicom_file_name_seg = os.path.join(patient_seg_output_dir, f'{patient_ID}_seg_slice_{j}.dcm')

                    # Create DICOM for image
                    save_dicom(image_slice, dicom_file_name_img, patient_ID, j)

                    # Optionally, create DICOM for segmentation
                    save_dicom(seg_slice, dicom_file_name_seg, patient_ID, j, segmentation=True)

                    # Save to JSON
                    entry = {
                        'ground_truth': dicom_file_name_img,
                        'observation': dicom_file_name_seg,
                        'Aorta_diss': Aorta_diss
                    }

                    with open(output_json_file, 'a') as json_file:
                        json.dump(entry, json_file, indent=4)