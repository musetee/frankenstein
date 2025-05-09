import os
IMG_EXTENSIONS = [
    #'.jpg', '.JPG', '.jpeg', '.JPEG',
    #'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 
    '.nrrd', '.nii.gz'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset_modality(dir, accepted_modalities = ["ct"], saved_name="source_filenames.txt"):
    # it works for root path of any layer:
    # data_path/Task1 or Task2/pelvis or brain
            # |-patient1
            #   |-ct.nii.gz
            #   |-mr.nii.gz
            # |-patient2
            #   |-ct.nii.gz
            #   |-mr.nii.gz
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for roots, _, files in sorted(os.walk(dir)): # os.walk digs all folders and subfolders in all layers of dir
        for file in files:
            if is_image_file(file) and file.split('.')[0] in accepted_modalities:
                path = os.path.join(roots, file)
                images.append(path)
    print(f'Found {len(images)} {accepted_modalities} files in {dir} \n')
    with open(saved_name,"w") as file:
        for image in images:
            file.write(f'{image} \n')
    return images

if __name__ == '__main__':
    path = "/gpfs/bwfor/work/ws/hd_qf295-foo/synthrad"
    #path = "/gpfs/bwfor/work/ws/hd_qf295-foo/synthrad/Task1/pelvis"
    images_list0=make_dataset_modality(path, ['ct'],"target_filenames.txt")
    images_list=make_dataset_modality(path, ['cbct', 'mr'],"source_filenames.txt")
    
    