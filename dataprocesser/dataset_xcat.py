import os
def list_img_pID_from_XCAT_folder(dir, saved_name="source_filenames.txt"):
    def is_image_file(filename):
        IMG_EXTENSIONS = [
                '.nrrd', '.nii.gz'
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
            if is_image_file(file):
                path = os.path.join(roots, file)
                patient_ID = os.path.basename(os.path.dirname(path))
                images.append(path)
                patient_IDs.append(patient_ID)
    print(f'Found {len(images)} files in {dir} \n')
    if saved_name is not None:
        with open(saved_name,"w") as file:
            for image in images:
                file.write(f'{image} \n')
    return images, patient_IDs
