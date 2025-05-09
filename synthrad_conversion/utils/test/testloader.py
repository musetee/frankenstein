from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
)
from torch.utils.data import DataLoader
import monai
import os
import glob
data_dir = r'E:\Projects\yang_proj\data\naeotomalpha\24032714_orange'
images = sorted(glob.glob(os.path.join(data_dir, 'br44', '*.nrrd')))
labels = sorted(glob.glob(os.path.join(data_dir, 'qr40_40kev', '*.nrrd')))
train_files = [{'image': image_name, 'label': label_name} for image_name, label_name in zip(images, labels)]
train_transforms = Compose([
    LoadImaged(image_only=True),
    EnsureChannelFirstd(image_only=True),
])
train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
for data in train_loader:
    print(data['image'].shape, data['label'].shape)
    break

