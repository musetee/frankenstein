import dicom2nifti
import SimpleITK as sitk
from dicom2nifti.exceptions import ConversionValidationError
from MsSeg.SegmentationNetworkBasis.NetworkBasis import image as Image
import os
import glob
import pydicom

#dataset_path = r"C:\Users\ms97\Documents\MRF-Daten\Messdaten"

def fromrootgroupconvert(dataset_path, nii_name='test'):
    i=0
    for patient_folder in glob.glob(dataset_path + "/*/"):
        print('-------------', patient_folder, '-------------')
        i=i+1
        t1_nii_path = os.path.join(dataset_path, patient_folder, f'{nii_name}_{i}.nii')
        try:
            try:
                t1_dicom_path = os.path.join(dataset_path, patient_folder)
                dicom2nifti.dicom_series_to_nifti(t1_dicom_path, t1_nii_path)
            except OSError as err:
                print("Finished for Sequence T1Map " + patient_folder)

            t1_img = sitk.ReadImage(t1_nii_path)
            data_info = Image.get_data_info(t1_img)
            print('Data Info T1:  ', data_info)
        except (KeyError, IndexError) as err:
            print("Failed for Sequence T1Map " + patient_folder + "  ", err)

def simplepatientconvert(patient_folder, nii_name='test'):
    t1_nii_path = os.path.join(patient_folder, f'{nii_name}.nii')
    try:
        try:
            dicom2nifti.dicom_series_to_nifti(patient_folder, t1_nii_path)
        except OSError as err:
            print("Finished for Sequence Dicom " + patient_folder)

        t1_img = sitk.ReadImage(t1_nii_path)
        data_info = Image.get_data_info(t1_img)
        print('Data Info T1:  ', data_info)
    except (KeyError, IndexError) as err:
        print("Failed for Sequence Dicom " + patient_folder + "  ", err)

def itkfromrootgroupconvert(dataset_path, nii_name='test'):
    i=0
    for patient_folder in glob.glob(dataset_path + "/*/"):
        print('-------------', patient_folder, '-------------')
        i=i+1
        try:
            try:
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(patient_folder)
                reader.SetFileNames(dicom_names)
                image = reader.Execute()
                basefoldername=os.path.basename(os.path.normpath(patient_folder))
                t1_nii_path = os.path.join(dataset_path, f'{basefoldername}.nii.gz')
                # Added a call to PermuteAxes to change the axes of the data
                image = sitk.PermuteAxes(image, [2, 1, 0])
                sitk.WriteImage(image, t1_nii_path)
            except OSError as err:
                print("Finished for Sequence T1Map " + patient_folder)

            t1_img = sitk.ReadImage(t1_nii_path)
            data_info = Image.get_data_info(t1_img)
            print('Data Info Dicom:  ', data_info)
        except (KeyError, IndexError) as err:
            print("Failed for Sequence Dicom " + patient_folder + "  ", err)

def itkforpatientconvert(patient_folder, nii_name='test'):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(patient_folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    t1_nii_path = os.path.join(patient_folder, f'{nii_name}.nii.gz')
    # Added a call to PermuteAxes to change the axes of the data
    image = sitk.PermuteAxes(image, [2, 1, 0])
    sitk.WriteImage(image, t1_nii_path)

if __name__=="__main__":
    dataset_path = r"D:\Data\dataNeaotomAlpha\DICOM_Naeotom\DICOM\23072115"
    itkfromrootgroupconvert(dataset_path)
    #patient_folder = r"D:\Data\dataNeaotomAlpha\Q0Q1Q4"
    #simpleitkconvert(patient_folder,'2511')
