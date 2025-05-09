function [scanData,pixelSpacing,sliceThickness,dicomInfo] = Read_Dicom_File(dataDir,scanDirNames)
% function to read in all .dcm,.ima or .nrrd files in a selected directory
% and write it into a single matrix A
scanFileDir = [dataDir '\' scanDirNames{1}]; % directory of the first scan file
[~,~,dataType] = fileparts(scanFileDir); % extract the file extension while neglecting other function outputs
switch dataType 
    case {'.dcm','.IMA',}
        [~,dataSize(1),dataSize(2)] = ReadDCM(scanFileDir); % extract size of dicom data array
        dicomInfo = dicominfo(scanFileDir);
        pixelSpacing = dicomInfo.PixelSpacing; % read out pixel spacing in mm
        sliceThickness = dicomInfo.SliceThickness;
        %get data from dicom file and read into 3D array
        scanData = zeros(dataSize(1),dataSize(2),numel(scanDirNames));
        for i = 1:numel(scanDirNames)
            fileDir = [dataDir '\' scanDirNames{i}];
            data_tmp = double(ReadDCM(fileDir));
            scanData(:,:,i) = data_tmp;
        end
    case '.nrrd'
        if length(scanDirNames) > 1
            tmp = nrrdread(scanFileDir);
            scanData = zeros(size(tmp,1),size(tmp,2),size(tmp,3),length(scanDirNames));
            for i = 1:length(scanDirNames)
                scanFileDir = [dataDir '\' scanDirNames{i}]; % directory of the scan file
                scanData(:,:,:,i) = nrrdread(scanFileDir);                
            end
        else
            scanData = double(nrrdread(scanFileDir));
        end
        pixelSpacing = [0.818359 0.818359]; % generate default pixel spacing in mm
        sliceThickness = 1.5; % generate default slice thickness in mm
        dicomInfo = {'none'};
end
end

