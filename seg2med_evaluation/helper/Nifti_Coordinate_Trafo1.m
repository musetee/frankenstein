function [trafImg] = Nifti_Coordinate_Trafo1(img)
% transforms the coordinate system of a nifti file in order to match dicom
% files, for segmentations, the image to be segmented has to be created
% from a dicom file
trafImg = permute(img,[2 1 3]); % swaps the x and y coordinates
trafImg = flip(trafImg,3); % mirrors the z dimension
end

