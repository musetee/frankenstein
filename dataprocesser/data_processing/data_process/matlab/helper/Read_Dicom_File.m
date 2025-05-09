function [scanData,pixelSpacing,sliceThickness,dicomInfo] = Read_Dicom_File(dataDir,dirNames)
% function to read in all .dcm,.ima or .nrrd files in a selected directory
% dataDir with names dirNames
% and write it into a single matrix A
scanFileDir = [dataDir '\' dirNames{1}]; % directory of the first scan file
[~,~,dataType] = fileparts(scanFileDir); % extract the file extension while neglecting other function outputs
switch dataType 
    case {'.dcm','.IMA'}
        [~,dataSize(2),dataSize(1)] = ReadDCM(scanFileDir); % extract size of dicom data array
        dicomInfo = dicominfo(scanFileDir);
        try
            pixelSpacing = dicomInfo.PixelSpacing; % read out pixel spacing in mm
            sliceThickness = dicomInfo.SliceThickness;
        catch
            pixelSpacing = NaN;
            sliceThickness = NaN;
        end
        %get data from dicom file and read into 3D array
        scanData = zeros(dataSize(1),dataSize(2),numel(dirNames));
        for i = 1:numel(dirNames)
            fileDir = [dataDir '\' dirNames{i}];
            data_tmp = double(ReadDCM(fileDir));
            try
                scanData(:,:,i) = data_tmp;
            catch
                scanData = NaN; 
            end
        end
    case '.nrrd'
        if length(dirNames) > 1
            % nrrdread is third party code
            tmp = nrrdread(scanFileDir);
            scanData = zeros(size(tmp,1),size(tmp,2),size(tmp,3),length(dirNames));
            for i = 1:length(dirNames)
                scanFileDir = [dataDir '\' dirNames{i}]; % directory of the scan file
                scanData(:,:,:,i) = nrrdread(scanFileDir);                
            end
        else
            scanData = double(nrrdread(scanFileDir));
        end
        pixelSpacing = [0.75 0.75]; % generate default pixel spacing in mm
        sliceThickness = 1.5; % generate default slice thickness in mm
        dicomInfo = {'none'};
end
end

