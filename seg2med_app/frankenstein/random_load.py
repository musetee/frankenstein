from seg2med_app.frankenstein.utils import *
import random
from tqdm import tqdm
def load_tissue_segmentation(patient_path):
    """
    优先加载三个单独的组织掩膜文件（subcutaneous_fat、torso_fat、skeletal_muscle），
    如果存在则按 label 值 1/2/3 组合；否则加载合并的 _seg_tissue.nii.gz 文件。
    """
    tissue_files = {
        "subcutaneous_fat": 1,
        "torso_fat": 2,
        "skeletal_muscle": 3
    }

    tissue_seg = None
    found_all = all(os.path.exists(os.path.join(patient_path, f"{name}.nii.gz")) for name in tissue_files)

    if found_all:
        print("✅ Found all 3 separate tissue files, loading them individually...")
        masks = []
        for name, label in tissue_files.items():
            path = os.path.join(patient_path, f"{name}.nii.gz")
            mask = nib.load(path).get_fdata().astype(np.uint8)
            mask = resize_volume_to_512(mask)
            mask = (mask > 0).astype(np.uint8) * label  # Ensure binary + assign label
            masks.append(mask)

        # Combine masks: take maximum value at overlapping positions
        tissue_seg = np.maximum.reduce(masks)

    else:
        print("⚠️ Separate tissue files not found, falling back to _seg_tissue.nii.gz...")
        for f in os.listdir(patient_path):
            if f.endswith("_seg_tissue.nii.gz"):
                tissue_seg = nib.load(os.path.join(patient_path, f)).get_fdata().astype(np.uint8)
                tissue_seg = resize_volume_to_512(tissue_seg)
                break

    return tissue_seg


def random_load(samples_path, needed_organs):
    patients = [f for f in os.listdir(samples_path) if os.path.isdir(os.path.join(samples_path, f))]
    chosen_patient = random.choice(patients)
    print('========chosen patient:', chosen_patient)
    patient_path = os.path.join(samples_path, chosen_patient)
    contour = nib.load(os.path.join(patient_path, "contour.nii.gz"))
    contour_affine = contour.affine
    contour_data = contour.get_fdata().astype(np.uint8)
    contour_data = resize_volume_to_512(contour_data)
    print("contour shape, ", contour_data.shape)
    progress_text = "⏳ random loading organs..."
    progress_bar = st.progress(0, text=progress_text)
    total_steps = len(needed_organs)
    
    step = 0
    organs = {}
    for organ in tqdm(needed_organs):
        step += 1
        organ_path = os.path.join(patient_path, "seg", f"{organ}.nii.gz")
        if os.path.exists(organ_path):
            organ_volume = nib.load(organ_path).get_fdata().astype(np.uint8)
            organs[organ] = resize_volume_to_512(organ_volume)
        if step == 1:
            print('original load volume shape:', organ_volume.shape)
        progress_bar.progress(step / total_steps, text=progress_text)
        
    tissue_seg = load_tissue_segmentation(patient_path)
    progress_bar.progress(1.0, text="✅ loading finished!")

    return contour_data, organs, tissue_seg

def random_load_chaotic(samples_path, editable_organs, fixed_organs, mode="default"):
    """
    根据加载模式，随机加载器官 + 轮廓 + 组织分割数据
    mode: "default" | "semi-chaotic" | "fully-chaotic"
    """
    patients = [f for f in os.listdir(samples_path) if os.path.isdir(os.path.join(samples_path, f))]
    
    def load_from_patient(patient_id, organs_to_load):
        """从某位病人中加载 contour、tissue_seg、特定器官"""
        path = os.path.join(samples_path, patient_id)
        loaded_organs = {}
        minimum_fixed_organ_slice_num = 99999
        for organ in organs_to_load:
            organ_path = os.path.join(path, "seg", f"{organ}.nii.gz")
            if os.path.exists(organ_path):
                organ_volume = nib.load(organ_path).get_fdata().astype(np.uint8)
                organ_slice_num = organ_volume.shape[2]
                if organ_slice_num < minimum_fixed_organ_slice_num:
                    minimum_fixed_organ_slice_num = organ_slice_num
                loaded_organs[organ] = resize_volume_to_512(organ_volume)
        return loaded_organs, minimum_fixed_organ_slice_num

    organ_sources = {}  # organ_name -> patient_id
    organs = {}
    
    if mode == "default":
        patient_id = random.choice(patients)
        st.info(f"🧍 Default Mode: all parts from a single  patient {patient_id}")
        contour = nib.load(os.path.join(samples_path, patient_id, "contour.nii.gz"))
        tissue_seg = load_tissue_segmentation(os.path.join(samples_path, patient_id))
        contour_data = resize_volume_to_512(contour.get_fdata().astype(np.uint8))
        
        all_organs = editable_organs + fixed_organs
        organs, minimum_fixed_organ_slice_num = load_from_patient(patient_id, all_organs)
        for organ in all_organs:
            organ_sources[organ] = patient_id

    elif mode == "semi-chaotic":
        fixed_patient = random.choice(patients)
        st.warning(f"🧪 Semi-Chaotic Mode: Contour, tissue segmentation, and fixed organs are loaded from a single patient {fixed_patient}, while editable organs are randomly mixed from different patients.")

        contour = nib.load(os.path.join(samples_path, fixed_patient, "contour.nii.gz"))
        tissue_seg = load_tissue_segmentation(os.path.join(samples_path, fixed_patient))
        contour_data = resize_volume_to_512(contour.get_fdata().astype(np.uint8))

        # fixed organs from one patient
        fixed_organs_loaded, minimum_fixed_organ_slice_num = load_from_patient(fixed_patient, fixed_organs)
        organs.update(fixed_organs_loaded)
        for organ in fixed_organs:
            organ_sources[organ] = fixed_patient

        # editable organs from random patients
        for organ in editable_organs:
            random_patient = random.choice(patients)
            path = os.path.join(samples_path, random_patient)
            organ_path = os.path.join(path, "seg", f"{organ}.nii.gz")
            if os.path.exists(organ_path):
                organ_volume = nib.load(organ_path).get_fdata().astype(np.uint8)
                organs[organ] = resize_volume_to_512(organ_volume)
                organ_sources[organ] = random_patient

    elif mode == "fully-chaotic":
        st.error("💥 fully-chaotic mode activated: each part is from a different patient. Extremely unstable! Use with caution.")

        contour_patient = random.choice(patients)
        contour = nib.load(os.path.join(samples_path, contour_patient, "contour.nii.gz"))
        contour_data = resize_volume_to_512(contour.get_fdata().astype(np.uint8))

        tissue_patient = random.choice(patients)
        tissue_seg = load_tissue_segmentation(os.path.join(samples_path, tissue_patient))

        minimum_fixed_organ_slice_num = 99999
        all_organs = editable_organs + fixed_organs
        for organ in all_organs:
            random_patient = random.choice(patients)
            organ_path = os.path.join(samples_path, random_patient, "seg", f"{organ}.nii.gz")
            if os.path.exists(organ_path):
                organ_volume = nib.load(organ_path).get_fdata().astype(np.uint8)
                organ_slice_num = organ_volume.shape[2]
                if organ_slice_num < minimum_fixed_organ_slice_num:
                    minimum_fixed_organ_slice_num = organ_slice_num
                organs[organ] = resize_volume_to_512(organ_volume)
                organ_sources[organ] = random_patient

    else:
        st.error(f"unknown mode: {mode}")
        return None, None, None, None
    st.session_state["orig_affine"] = contour.affine
    return contour_data, organs, tissue_seg, organ_sources, minimum_fixed_organ_slice_num