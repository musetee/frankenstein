function [trafImg] = Nifti_Coordinate_Trafo2(img)
% transforms the coordinate system of a nifti file containing the segmentation of generated CT data in order to match dicom
% files, for segmentations, the image to be segmented has to be created
% from a nrrd file
trafImg = permute(img,[2 1 3]); % swaps the x and y coordinates
trafImg = flip(trafImg,2); % mirrors the y dimension
trafImg = flip(trafImg,1); % mirrors the x dimension
end

