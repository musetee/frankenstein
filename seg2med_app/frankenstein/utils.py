import numpy as np
import cv2
import os
import json
from io import BytesIO
from PIL import Image
import nibabel as nib
from seg2med_app.app_utils.image_utils import default_plt_origin_type
# ----------------------------
# 显示切片函数 (带颜色映射)
# ----------------------------
def get_slice_img(data, organ_name, idx=None):
    num_slices = data.shape[2]
    if idx == None:
        idx = num_slices // 2
    if organ_name == 'seg_tissue':
        data = data.astype(np.uint8)
        data[data == 1] = 201
        data[data == 2] = 202
        data[data == 3] = 203 # 201, 202, 203 before contour merge
    elif organ_name == 'contour':
        data = data
    slice_img = np.transpose(data[:, :, idx])
    return slice_img

def resize_volume_to_512(volume, new_H= 512, new_W= 512):
    """
    将输入 volume (H, W, D) 调整为 512×512×D，通过 crop or pad。
    """
    H, W, D = volume.shape
    resized = np.zeros((new_H, new_W, D), dtype=volume.dtype)

    pad_top = max((new_H - H) // 2, 0)
    pad_left = max((new_W - W) // 2, 0)

    crop_top = max((H - new_H) // 2, 0)
    crop_left = max((W - new_W) // 2, 0)

    for z in range(D):
        slice_2d = volume[:, :, z]

        # crop if needed
        if H > new_H:
            slice_2d = slice_2d[crop_top:crop_top + new_H, :]
        if W > new_W:
            slice_2d = slice_2d[:, crop_left:crop_left + new_W]

        # pad if needed
        h_pad = new_H - slice_2d.shape[0]
        w_pad = new_W - slice_2d.shape[1]
        slice_2d = np.pad(slice_2d,
                          ((h_pad // 2, h_pad - h_pad // 2),
                           (w_pad // 2, w_pad - w_pad // 2)),
                          mode='constant', constant_values=0)

        resized[:, :, z] = slice_2d

    return resized


def convert_organ_mask_to_label(slice_img, organ_name, organ_to_label):
    if organ_name == 'seg_tissue':
        slice_img[slice_img == 1] = 201 # tombine contour here
        slice_img[slice_img == 2] = 202
        slice_img[slice_img == 3] = 203
        mask = slice_img.astype(np.uint8)
    elif organ_name == 'contour':
        mask = slice_img.astype(np.uint8)
    else:
        label = organ_to_label[organ_name]
        slice_img[slice_img > 0] = label
        mask = slice_img.astype(np.uint8)
    return mask

def convert_label_to_rgb_map(mask, label_to_colors):
    rgb_map = np.zeros((*mask.shape, 3), dtype=np.uint8)

    for label, rgb_str in label_to_colors.items():
        rgb_vals = [int(v) for v in rgb_str.split(",")]
        label = int(label)  # 保证类型一致
        mask_label = (mask == label)
        for c in range(3):
            rgb_map[:, :, c][mask_label] = rgb_vals[c]
    
    return rgb_map

def show_organ_axial(slice_img, organ_name, label_to_colors=None, organ_to_label=None, show_img=True):
    #print(f'organ_name min and max: {np.min(data)}, {np.max(data)}')
    if label_to_colors:
        label_mask = convert_organ_mask_to_label(slice_img, organ_name, organ_to_label)
        rgb_map = convert_label_to_rgb_map(label_mask, label_to_colors)

    else:
        rgb_map = (slice_img > 0).astype(np.uint8)  #* 255
    
    if show_img:
        '''fig, ax = plt.subplots()
        ax.imshow(rgb_map, origin=default_plt_origin_type)
        ax.axis('off')
        st.pyplot(fig)'''
        st.image(Image.fromarray(rgb_map), caption=f"{organ_name}", use_column_width=True)
        
    return label_mask, rgb_map

# ----------------------------
# 保存为 NIfTI
# ----------------------------
def save_custom_slice(output_np_array, affine):
    img = nib.Nifti1Image(output_np_array, affine)
    bio = BytesIO()
    nib.save(img, bio)
    bio.seek(0)
    return bio



import cv2

def mask_to_path_object(mask2d, color_rgba, organ_name, selectable=False):
    contours, _ = cv2.findContours(mask2d.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for contour in contours:
        if cv2.contourArea(contour) < 5:
            continue
        path = "M " + " L ".join([f"{pt[0][0]} {pt[0][1]}" for pt in contour])
        path += " Z"
        #print(path)
        objects.append({
            "type": "path",
            "path": path,
            "fill": color_rgba,
            "stroke": "#000000",
            "strokeWidth": 1,
            "name": organ_name,
            "selectable": selectable,
            "evented": selectable,
            "hasControls": selectable,
            "lockMovementX": not selectable,
            "lockMovementY": not selectable
        })
    return objects
    
import json
import streamlit as st
import streamlit.components.v1 as components

def render_fabric_canvas(drawing_objects, canvas_size=512):
    
    with open(os.path.join(st.session_state["app_root"], "frankenstein","organ_editor.html"), "r", encoding="utf-8") as f:
        html_template = f.read()

    fabric_data_json = json.dumps({"objects": drawing_objects})

    # 注入 JSON 到 HTML 中
    
    html_filled = html_template.replace("{{ data }}", fabric_data_json)
    components.html(html_filled, height=canvas_size+100, width=canvas_size+100)

import hashlib
def hash_state(state_dict):
    """将字典内容转为字符串后生成hash"""
    state_json = json.dumps(state_dict, sort_keys=True)
    return hashlib.md5(state_json.encode()).hexdigest()

import datetime
def hash_current_time():
    now_str = datetime.datetime.now().isoformat()  # 形如 '2025-05-07T15:30:12.123456'
    return hashlib.md5(now_str.encode()).hexdigest()

import hashlib
import json

def hash_slice_indices(indices_dict):
    json_str = json.dumps(indices_dict, sort_keys=True)
    return hashlib.md5(json_str.encode()).hexdigest()

import math
def canvas_json_to_label_mask(json_objects, canvas_shape, label_color_map, organ_to_label):
    """
    将 streamlit-drawable-canvas 返回的 JSON objects 转换为 label 掩膜图

    参数:
    - json_objects: canvas_result.json_data["objects"]
    - canvas_shape: (H, W)
    - organ_to_label: dict[str → int]

    返回:
    - label_mask: np.ndarray with shape (H, W), 每个像素为器官 label（int）
    """
    debug = False
    H, W = canvas_shape
    label_mask = np.zeros((H, W), dtype=np.uint16)
    
    contour_label_mask = np.zeros((H, W), dtype=np.uint16)
    organ_label_mask = np.zeros((H, W), dtype=np.uint16)
    # 反向映射：颜色字符串 -> organ 名
    color_to_label_map = {v: k for k, v in label_color_map.items()}
    for obj in json_objects:
        rgba = obj.get("fill", "")  # e.g., "rgba(255,0,0,0.4)"
        rgb = rgba.replace("rgba(", "").split(")")[0].split(",")[:3]
        rgb_str = ",".join(str(int(float(v))) for v in rgb)
        if rgb_str == '255,255,255':
            print('detect contour')
            label_value = 1 # contour
        else:
            label_value = color_to_label_map.get(rgb_str, None)
            #label_value += 1
        #print('get label value ', label_value, 'from rgb', rgb_str)
        # 获取 path 点列表（需要偏移、缩放等）
        if obj.get("type") == "path" and "path" in obj:
            
            left = obj.get("left", 0)
            top = obj.get("top", 0)
            scaleX = obj.get("scaleX", 1.0)
            scaleY = obj.get("scaleY", 1.0)
            angle = obj.get("angle", 0.0)  # degrees

            width = obj.get("width", 0)
            height = obj.get("height", 0)
            
            path_raw_points = []
            for cmd in obj["path"]:
                if cmd[0] in ("M", "L"):
                    path_raw_points.append([float(cmd[1]), float(cmd[2])])
            path_raw_points = np.array(path_raw_points)

            # the origin (left top) point of bbox, will not change during transformation
            path_min_x = np.min(path_raw_points[:, 0])
            path_min_y = np.min(path_raw_points[:, 1])
            
            path_max_x = np.max(path_raw_points[:, 0])
            path_max_y = np.max(path_raw_points[:, 1])
            
            # width = path_max_x - path_min_x
            # height = path_max_y - path_min_y
            if debug:
                print('')
                print('current object information:')
                print('left:', obj.get("left", 0))
                print('top:', obj.get("top", 0))
                print('width:', obj.get("width", 0))
                print('height:', obj.get("height", 0))
                print('angle:', obj.get("angle", 0))
                print('scaleX:', obj.get("scaleX", 1.0))
                print('scaleY:', obj.get("scaleY", 1.0))
                print('skewX:', obj.get("skewX", 1.0))
                print('skeweY:', obj.get("skeweY", 1.0))
                print('path_min_x:', path_min_x)
                print('path_max_x:', path_max_x)
                print('check: width = path_max_x - path_min_x', width, path_max_x - path_min_x)
                print('check: height = path_max_y - path_min_y', height, path_max_y - path_min_y)
                print('path_min_y:', path_min_y)
                print('path_max_y:', path_max_y)
            
            # try to rotate with 0
            #cx = 256
            #cy = 256
            theta = angle * math.pi / 180.0  # 逆时针旋转角度（fabric 是顺时针）
            
            xlist = []
            ylist = []
            x0list = []
            y0list = []
            
            path_points = []
            for x, y in path_raw_points:
                # 去掉 path 内部的偏移
                x0 = (x - path_min_x) * scaleX
                y0 = (y - path_min_y) * scaleY

                xlist.append(x)
                ylist.append(y)
                x0list.append(x0)
                y0list.append(y0)
                
                # calculate bbox center
                cx = (path_max_x - path_min_x) / 2 * scaleX 
                cy = (path_max_y - path_min_y) / 2 * scaleY 
                
                # box rotation
                box_lefttop_x = -cx
                box_lefttop_y = -cy
                box_lefttop_x_rot = box_lefttop_x * np.cos(theta) - box_lefttop_y * np.sin(theta)
                box_lefttop_y_rot = box_lefttop_x * np.sin(theta) + box_lefttop_y * np.cos(theta)
                
                x_shift = left - box_lefttop_x_rot
                y_shift = top - box_lefttop_y_rot
                
                # point rotation
                x1 = x0 - cx
                y1 = y0 - cy
                x_rot = x1 * np.cos(theta) - y1 * np.sin(theta)
                y_rot = x1 * np.sin(theta) + y1 * np.cos(theta)

                # 平移到画布坐标
                x_final = x_shift + x_rot # left + cx +  
                y_final = y_shift + y_rot # top + cy +  
                path_points.append([x_final, y_final])
            
            #print('min max of x0:', np.min(x0list), np.max(y0list))  
            if debug:
                print('min of x0, y0:', np.min(x0list), np.min(y0list))   
                print('max of x0, y0:', np.max(x0list), np.max(y0list))   
            path_points = np.array([path_points], dtype=np.int32)

            # 用 OpenCV 绘制多边形掩膜
            cv2.fillPoly(label_mask, path_points, color=label_value)
            
            if rgb_str == '255,255,255':
                cv2.fillPoly(contour_label_mask, path_points, color=label_value)
            else:
                cv2.fillPoly(organ_label_mask, path_points, color=label_value)
    return label_mask, contour_label_mask, organ_label_mask

import numpy as np
import matplotlib.pyplot as plt
import io
    
def normalize_label_mask_log_enhanced(mask, base=1.1):
    mask = mask.astype(np.float32).copy()

    # 用 nan 屏蔽 0
    log_mask = np.where(mask == 0, np.nan, np.log(mask + 1e-5) / np.log(base))

    # 将 label==1 的位置设为固定亮度（例如 30）
    out = np.zeros_like(mask, dtype=np.uint8)
    out[mask == 1] = 50  # 稍微亮一点

    # 对其余非 0、非 1 的部分 log 映射到 50–255
    dynamic_mask = np.nan_to_num(log_mask, nan=0.0)
    dynamic_mask[mask <= 1] = 0  # 不动 0 和 1

    if dynamic_mask.max() > 0:
        norm_dynamic = 205 * (dynamic_mask / dynamic_mask.max()) + 50  # 映射到 [50,255]
        out[mask > 1] = norm_dynamic[mask > 1].astype(np.uint8)

    return out


def normalize_grayscale_for_display(mask):
    """
    将 label mask 映射到 0–255 之间的灰度值（便于 st.image 正确显示）
    """
    mask = mask.astype(np.float32)
    if mask.max() == mask.min():
        return np.zeros_like(mask, dtype=np.uint8)  # 避免除以 0
    norm = 255 * (mask - mask.min()) / (mask.max() - mask.min())
    return norm.astype(np.uint8)