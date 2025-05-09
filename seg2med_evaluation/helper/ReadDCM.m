function [arrayHU, dcmWidth, dcmHeight] = ReadDCM(dcmFile)
% reads a '.dcm' or '.ima' file
% returns 
% arrayHU = an array with the image pixels in HU
% dcmWidth = width of image matrix
% dcmHeight = width of image matrix
dcmHeader = dicominfo(dcmFile); %reads in the dicom header of the first file
dcmData = double(dicomread(dcmFile)); %reads Dicom data as a double variable (default = uint)
arrayHU = dcmHeader.RescaleSlope * dcmData + dcmHeader.RescaleIntercept; %converts data values into Hounsfield Unites (HU)
dcmWidth = double(dcmHeader.Width);
dcmHeight = double(dcmHeader.Height);
end