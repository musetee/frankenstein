
import os
def list_img_pID_from_LITS_folder(dir, isseg=False, saved_name=None):
    def is_image_file(filename):
        IMG_EXTENSIONS = [
                '.nrrd', '.nii.gz', '.nii'
            ]
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    # it works for root path of any layer:
    # data_path/Task1 or Task2/pelvis or brain
            # |-patient1
            #   |-ct.nii.gz
            #   |-mr.nii.gz
            # |-patient2
            #   |-ct.nii.gz
            #   |-mr.nii.gz
    images = []
    patient_IDs = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for roots, _, files in sorted(os.walk(dir)): # os.walk digs all folders and subfolders in all layers of dir
        for file in files:
            path = os.path.join(roots, file)
            patient_ID = os.path.splitext(file)[0]
            if isseg is True:
                if "seg" in file:
                    images.append(path)
                    patient_IDs.append(patient_ID)
            else: 
                if "seg" not in file:
                    images.append(path)
                    patient_IDs.append(patient_ID)
    print(f'Found {len(images)} files in {dir} \n')
    if saved_name is not None:
        with open(saved_name,"w") as file:
            for image in images:
                file.write(f'{image} \n')
    return images, patient_IDs
