from dataprocesser.preprocess_MR.step3_vibe_resetsignal import calculate_signal_GRE_T1, calculate_signal_T2_SPACE, calculate_signal_vibe, calculate_signal_vibe_opp, calculate_signal_vibe_dixon
from dataprocesser.simulation_functions import _create_body_contour, _merge_seg_tissue, _merge_seg_contour, _assign_value_ct, _assign_value_mr, _create_body_contour_by_tissue_seg
from dataprocesser.Preprocess_MRCT_mask_conversion import convert_segmentation_mask

import nibabel as nib
import numpy as np
def simulate_modality(contour, output_modality, merged_seg_tissue, input_modality=None,
                   csv_file_ct = r'params_ct.csv',
                   csv_file_mr = r'params_mr.csv',
                   custom_signal_fn=None,):
    
    
    modality_config = {
        0: {
            'body_threshold': -500,
            'assign_fn': _assign_value_ct,
            'csv_file': csv_file_ct,
            'signal_fn': None
        },
        1: { # t1 weighted gradient echo
            'body_threshold': 100,
            'assign_fn': _assign_value_mr,
            'csv_file': csv_file_mr,
            'signal_fn': calculate_signal_GRE_T1
        },
        2: { # t2 space
            'body_threshold': 10,
            'assign_fn': _assign_value_mr,
            'csv_file': csv_file_mr,
            'signal_fn': calculate_signal_T2_SPACE
        },
        3: { # vibe in
            'body_threshold': 30,
            'assign_fn': _assign_value_mr,
            'csv_file': csv_file_mr,
            'signal_fn': calculate_signal_vibe
        },
        4: { # vibe opp
            'body_threshold': 30,
            'assign_fn': _assign_value_mr,
            'csv_file': csv_file_mr,
            'signal_fn': calculate_signal_vibe_opp
        },
        5: { # vibe dixon
            'body_threshold': 30,
            'assign_fn': _assign_value_mr,
            'csv_file': csv_file_mr,
            'signal_fn': calculate_signal_vibe_dixon
        }
    }
    
    if output_modality == 10000 and custom_signal_fn is not None:
        output_config = {
            'body_threshold': 30,
            'assign_fn': _assign_value_mr,
            'csv_file': csv_file_mr,
            'signal_fn': custom_signal_fn
        }
        print('use customised sequence')
    else:
        output_config = modality_config.get(output_modality)
    
    input_config = modality_config.get(input_modality)
    if output_config is None:
        raise ValueError(f"Unsupported modality: {output_modality}")

    if input_modality == 'ct' and output_modality not in [0]:
        print('convert CT mask → MR mask')
        merged_seg_tissue = convert_segmentation_mask(merged_seg_tissue, 
                                                            source_csv=csv_file_ct,
                                                            target_csv=csv_file_mr,
                                                            body_contour_value=1000)
        
    elif input_modality == 'mr' and output_modality not in [1, 2, 3, 4, 5, 10000]:
        print('convert MR mask → CT mask')
        merged_seg_tissue = convert_segmentation_mask(merged_seg_tissue, 
                                                            source_csv=csv_file_mr,
                                                            target_csv=csv_file_ct,
                                                            body_contour_value=1000)
        
    
    merged_seg_tissue_contour = _merge_seg_contour(merged_seg_tissue, contour)
    print("values in merged_seg_tissue_contour", np.unique(merged_seg_tissue_contour))
    merged_seg_tissue[merged_seg_tissue==1000] = 1
    merged_seg_tissue[merged_seg_tissue==1001] = 1

    # 指定 assign 函数处理 CT 或 MR
    if output_config['signal_fn'] is not None:
        prior_image = output_config['assign_fn'](output_config['csv_file'], merged_seg_tissue_contour, output_config['signal_fn'])
    else:
        prior_image = output_config['assign_fn'](output_config['csv_file'], merged_seg_tissue_contour)

    #processed_img = nib.Nifti1Image(prior_image, target_affine)
    
    return prior_image