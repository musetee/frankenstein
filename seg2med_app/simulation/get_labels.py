import pandas as pd
import os
def get_labels(seg_max, app_root):
    if seg_max>60:
        print(f'max number in segmentation is {seg_max}, recognized as ct')
        label_map = pd.read_csv(os.path.join(app_root, "simulation/params_ct.csv"))  # 👉 CT 标签定义（你手动准备）
        input_modality = 'ct'
    else:
        print(f'max number in segmentation is {seg_max}, recognized as mr')
        label_map = pd.read_csv(os.path.join(app_root, "simulation/params_mr.csv"))  # 👉 MR 标签定义（你手动准备）
        input_modality = 'mr'
    
    name_dict = "Organ Name" # "name"
    id_dict = "Order Number" #"id"
    label_map = label_map[label_map[id_dict] != 0]  # remove first row, with 0=body contour
    organ_to_label = dict(zip(label_map[name_dict], label_map[id_dict]))
    label_to_organ = dict(zip(label_map[id_dict], label_map[name_dict]))  # 反向映射
    label_ids = label_map[id_dict].tolist()

    return organ_to_label, label_to_organ, label_ids, input_modality