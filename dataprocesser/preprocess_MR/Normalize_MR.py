import numpy as np
import SimpleITK as sitk
import pandas as pd
import os
import tensorflow as tf


def normalize(img, vmin_out=0, vmax_out=1, norm_min_v=None, norm_max_v=None, epsilon = 1e-5):
    if norm_min_v == None and norm_max_v == None:
        try:
            norm_min_v = tf.reduce_min(img)
            norm_max_v = tf.reduce_max(img)
        except:
            norm_min_v = np.min(img)
            norm_max_v = np.max(img)
    else:
        flags = img < norm_min_v
        img[flags] = norm_min_v
        flags = img > norm_max_v
        img[flags] = norm_max_v

    #normalize to interval [0,1)
    img = (img - norm_min_v) / (norm_max_v - norm_min_v + epsilon)

    #normalize to interval [vmin_out, vmax_out)
    img = img * (vmax_out - vmin_out) + vmin_out

    return img



filepath = r"E:\XCATproject\SynthRad_GAN\trainingdata_new\MR_VIBE_Supplementary_nrrd_73"


for root, dirs, files in os.walk(filepath):
    for filename in files:
        print(filename)
        mr = sitk.ReadImage(os.path.join(filepath, filename))
        mr_array = sitk.GetArrayFromImage(mr)

        savepath = os.path.join(r"E:\XCATproject\SynthRad_GAN\trainingdata_new\MR_VIBE_Supplementary_nrrd_73_p10_90_norm0_255", filename)

        mr_array_normalized = normalize(mr_array, 0, 255, np.percentile(mr_array, 10), np.percentile(mr_array, 90),
                                        epsilon=0)

        mr_normalized = sitk.GetImageFromArray(mr_array_normalized)

        mr_normalized.SetSpacing(mr.GetSpacing())
        mr_normalized.SetOrigin(mr.GetOrigin())
        mr_normalized.SetDirection(mr.GetDirection())

        sitk.WriteImage(mr_normalized, savepath)