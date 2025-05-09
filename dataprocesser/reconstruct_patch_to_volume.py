import torch
def initialize_collection(first_data):
    collected_patches = []
    collected_coords = []
    #first_data = next(iter(train_loader))
    original_spatial_shape = first_data['original_spatial_shape']
    data_patch_0 = first_data['img']
    #print(data_patch_0.meta['filename_or_obj'])
    volume_shape = tuple(torch.max(dim_shape).item() for dim_shape in original_spatial_shape)
    reconstructed_volume = torch.zeros(volume_shape, dtype=data_patch_0.dtype)
    print('empty volume_shape:',volume_shape)
    # Initialize a volume to keep count of the number of patches added at each location
    count_volume = torch.zeros(volume_shape, dtype=torch.int)
    return collected_patches, collected_coords, reconstructed_volume, count_volume

def reconstruct_volume(collected_patches, collected_coords, reconstructed_volume, count_volume=None):
    A_data = collected_patches[0]
    batch_size = A_data.shape[0]
    batch_num = len(collected_patches)
    print('batch_num:',batch_num)
    for data_idx in range(batch_num):
        data = collected_patches[data_idx]
        patch_coords = collected_coords[data_idx]
        #print(patch_coords)
        for batch_idx in range(batch_size):
            data_patch_idx = data[batch_idx]
            patch_coords_idx = patch_coords[batch_idx]
            channel_start, channel_end = patch_coords_idx[0]
            x_start, x_end = patch_coords_idx[1]
            y_start, y_end = patch_coords_idx[2]
            z_start, z_end = patch_coords_idx[3]
            
            # Place the patch in the reconstructed volume
            try:
                reconstructed_volume[x_start:x_end, y_start:y_end, z_start:z_end] = data_patch_idx[0]
                if count_volume is not None:
                    count_volume[x_start:x_end, y_start:y_end, z_start:z_end] = 1
            except IndexError as e:
                print(f"IndexError: {e} - check patch coordinates and dimensions")
                print('patch_coords_idx:',patch_coords_idx)
                print('data shape:',data_patch_idx.shape)
                print('to fill shape:',reconstructed_volume[x_start:x_end, y_start:y_end, z_start:z_end].shape)
                print('check the div_size and patch_size, they should be at least the same')
            '''
            si_input(B_data[batch_idx])
            si_seg(A_data[batch_idx])
            grad=gradient_calc(B_data[batch_idx])
            si_grad(grad)
            '''
            # Avoid division by zero
            #count_volume = torch.where(count_volume == 0, torch.ones_like(count_volume), count_volume)
            
            # Average out the overlapping areas
            #reconstructed_volume = reconstructed_volume / count_volume
    return reconstructed_volume, count_volume