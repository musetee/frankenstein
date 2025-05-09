% script to evaluate a network for generating synthetic CT images from XCAT
% phantom data
genDir = 'A:/MIDL/Dominik_MIDL/genCT'; % directory containing network directories with generated CT data
genList = dir(genDir); % list all files in directory
genNames = {genList(3:end).name}; % list all the filnames
nrNetworks = numel(genNames); % number of networks
%% 
%read in XCAT data
XCATDataDir = 'A:/MIDL/Dominik_MIDL/XCAT'; % directory of XCAT data used for CT generation
XCATListDir = dir(XCATDataDir); % list all files in selected directory
XCATDirNames = {XCATListDir(3:end).name}; % list all the filnames
nrXCATPhantoms = numel(XCATDirNames); % number of phantoms in directory
%%
% initialize parameters to evaluate
NoiseEvaluation = 0; % are the noise properties to be evaluated?
ReferenceNPSEvaluation = 0; % patient NPS to be evaluated?
EdgeEvaluation = 1; % are the edge and feature properties to be evaluated?
NiftiFromDicom = 1; % were the reference segmentation masks extracted from nii files converted from dicom files? Put 0 for .nrrd
DicomCTImages = 1; % were the reference patient images dicom files? Put 0 for .nrrd
%%
if EdgeEvaluation
    [Result.FSIM,Result.SSIM,Result.MAE,Result.FaultyPixel,Result.EPR,Result.EdgeGenRatio,Result.GMSdev] = deal(cell(nrXCATPhantoms,1)); %,Result.MAD,Result.NMSE
    [Data.PCBP,Data.MAE,Data.SSIM,Data.FSIM,Data.GMSD,Data.EGR] = deal(cell(nrXCATPhantoms,nrNetworks)); % container for statistics of all metrics
end
%%
if NoiseEvaluation
    %% 
    % initiale cells for metrics
    [Result.NPSAccuracy,Result.NPSCorrelation,Result.NoiseMagnitude,Result.MaxRatio] = deal(cell(nrXCATPhantoms,1));
    [Data.NM,Data.PCC,Data.MPE,Data.MMR] = deal(cell(nrXCATPhantoms,nrNetworks));
    %%
    % read segmentations of generated data
    genSegmDir = 'A:/Dominik_MIDL/genSegm'; % directories containing generated CT data
    %%
    if ReferenceNPSEvaluation
        % read in training and segmentation data to calculate real noise properties for comparison
        segmDir = 'A:/Dominik_MIDL/Patsegm'; % directories containing segmentation of real patient data
        segmList = dir(segmDir);
        segmNames = {segmList(3:end).name};
        %%
        % read patient data
%         patDir = 'D:\Russ\CT_Images\synCT\Patient'; % directories containing real patient data
        patDir = 'A:/Dominik_MIDL/Patnrrd'; % directories containing real patient data
        patList = dir(patDir);
        patNames = {patList(3:end).name};
        [Data.ReferenceNPS2D,Data.ReferenceFreq,Data.ReferenceNPScurves,Data.LiverValues] = deal(cell(numel(segmNames),1));
        %%
        % segment patient data and calculate NPS
        for i = 1:numel(segmNames)
            patScanDir = [patDir '\' patNames{i}];
            patScanList = dir(patScanDir);
            patScanNames = {patScanList(3:end).name};
            if DicomCTImages
                % sortn is third party code
                patScanNames = sortn(patScanNames); % sort the cell array in natural order, if they are dicom images
            end
            [patData,patPixelSpacing] = Read_Dicom_File(patScanDir,patScanNames);
            segmData = double(niftiread([segmDir '\' segmNames{i}]));
            if NiftiFromDicom
                segmData = Nifti_Coordinate_Trafo1(segmData); % transform the coordinate system for correct representation, depends, whether the image to be segmented was created from a nrrd file (2) or dicom file (1)
            else
                segmData = Nifti_Coordinate_Trafo2(segmData); % transform the coordinate system for correct representation, depends, whether the image to be segmented was created from a nrrd file (2) or dicom file (1)
            end    
            liverData = segmData.*patData;
            Data.ReferenceNPS2D{i} = patPixelSpacing(1)*patPixelSpacing(2)*FFT_Segmented_Noise(liverData);
            [Data.ReferenceFreq{i},Data.ReferenceNPScurves{i}] = Radial_from_2D_NPS(Data.ReferenceNPS2D{i},patPixelSpacing(1),1);   
            % calculate reference STD
            liverData(segmData==0) = NaN;
            Data.LiverSTD{i} = nanstd(liverData(:));
        end
        %%
        % compute convex hull of radial NPS in order to create a valid
        % variable space for generated NPS
        refNPS = vertcat(Data.ReferenceNPScurves{:})';
        Data.ReferenceNPS = mean(refNPS,2)';
        Data.minNPS = min(refNPS,[],2);
        Data.maxNPS = max(refNPS,[],2);
        Data.LiverNoiseMean = mean(cell2mat(Data.LiverSTD(:)));
        Data.LiverNoiseSTD = std(cell2mat(Data.LiverSTD(:)));
    end
end
clear refNPS genList XCATListDir XCATListDir segmList i liverData patData patDir patList patNames patPixelSpacing patScanDir patScanList patScanNames segmData segmDir segmNames
%%
for k = 1:nrNetworks
    %%
    % read in XCAT stuff for prepping
    [XCATData,XCATPixelSpacing,XCATSliceThickness] = Read_Dicom_File(XCATDataDir,XCATDirNames(1)); % read in first file to extract pixel spacing and slice thickness
    [nrXCATRows,nrXCATColumns,nrXCATSlices] = size(XCATData); % extract size of generated image matrix
    %%
    % read in generated image stuff
    network = genNames{k};
    genDataDir = [genDir '\' network];
    genListDir = dir(genDataDir); % list all files in selected directory
    genDirNames = {genListDir(3:end).name}; % list all the filnames
    nrGenPhantoms = numel(genDirNames); % number of phantoms in directory
    genData = Read_Dicom_File(genDataDir,genDirNames(1)); % read in first file to extract pixel spacing and slice thickness
    [nrGenRows,nrGenColumns,nrGenSlices] = size(genData); % extract size of generated image matrix
    %%
    % compare generated and XCAT data size and return error if not matching
    if nrXCATRows ~= nrGenRows || nrXCATColumns ~= nrGenColumns
        error('Size of XCAT data and generated data not matching')
    elseif nrXCATPhantoms ~= nrGenPhantoms
        error('Number of Phantoms in XCAT and generated data not matching')        
    end
    n = nrXCATSlices - nrGenSlices; % parameter depending on how many slices were used for training the networks    
    clear nrGenPhantoms nrXCATSlices XCATData nrGenRows nrGenColumns genData genListDir nrXCATRows nrXCATColumns
    %%
    for i = 1:nrXCATPhantoms
        %%
        phantom = genDirNames{i};
        genData = Read_Dicom_File(genDataDir,{phantom}); % read in generated data
        XCATData = Read_Dicom_File(XCATDataDir,XCATDirNames(i)); % read in XCAT data
        XCATData = XCATData(:,:,1+n/2:end-n/2); % first and last n/2 slices are not in synthetic CTs
        %%
        if EdgeEvaluation
            XCATContour = XCATData ~= -1000; % extract XCAT body contour, always 0 outside the body
            compXCATContour = imcomplement(XCATContour); % inverts the logical binary mask
            segmData = XCATContour.*genData;
            segmData(XCATContour==0) = -1000;
            negData = compXCATContour.*genData;
            negData(compXCATContour==0) = -2000;
            absXCAT = XCATData; % create new XCAT data with -2000 outside of the body contour for better excluding it while absolute comparison
            absXCAT(XCATContour==0) = -2000;
            absSegmData = segmData; % create new segmented data with -2000 outside of the body contour for better excluding it while absolute comparison
            absSegmData(XCATContour==0) = -2000;
            [FSIMtmp,SSIMtmp,MAEtmp,FaultyPixeltmp,EPRtmp,EdgeGenRatiotmp,GMSdevtmp] = deal(zeros(nrGenSlices,1)); %,MADtmp,NMSEtmp
            for j = 1:nrGenSlices
                % read in single axial slices
                gentmp = segmData(:,:,j);
                absGentmp = absSegmData(:,:,j);
                XCATtmp = XCATData(:,:,j);
                absXCATtmp = absXCAT(:,:,j);
                negDatatmp = negData(:,:,j);
                % evaluate the slices
                FSIMtmp(j) = FeatureSIM(rescale(XCATtmp,0,255),rescale(gentmp,0,255)); % calculates the feature similarity index
                SSIMtmp(j) = ssim(rescale(gentmp),rescale(XCATtmp)); % calculates the structural similarity index
                MAEtmp(j) = Mean_Absolute_Error(absXCATtmp,absGentmp); % calculates the mean absolute error
%                 NMSEtmp(j) = Normalized_Mean_Square_Error(absXCATtmp,absGentmp); % calculates the normalized mean square error
                FaultyPixeltmp(j) = Faulty_Pixel(negDatatmp); % calculates the number of 'faulty pixels in the generated image
                [EPRtmp(j),EdgeGenRatiotmp(j)] = Edge_Preservation_Ratio(XCATtmp,gentmp); % calculates the edge preservation ratio and
                GMSdevtmp(j) = Gradient_Magnitude_Similarity_Deviation(XCATtmp,gentmp); % calculates the gradient magnitude similarity deviation
%                 tmp = MAD_index(XCATtmp,gentmp); % calculates the most apparent distortion index
%                 MADtmp(j) = tmp.MAD;
            end
            Result.FSIM{i} = FSIMtmp;
            Data.FSIM{i,k} = FSIMtmp;
            Result.SSIM{i} = SSIMtmp;
            Data.SSIM{i,k} = SSIMtmp;
            Result.MAE{i} = MAEtmp;
            Data.MAE{i,k} = MAEtmp;
%             Result.NMSE{i} = NMSEtmp;
            Result.FaultyPixel{i} = FaultyPixeltmp;
            Data.PCBP{i,k} = FaultyPixeltmp;
            Result.EPR{i} = EPRtmp;
            Result.EdgeGenRatio{i} = EdgeGenRatiotmp;
            Data.EGR{i,k} = EdgeGenRatiotmp;
            Result.GMSdev{i} = GMSdevtmp;
            Data.GMSD{i,k} = GMSdevtmp;
%             Result.MAD{i} = MADtmp;
            clear absXCATtmp absGentmp absSegmData absXCAT tmp i j genData XCATData XCATContour compXCATContour segmData negData gentmp XCATtmp negDatatmp FSIMtmp SSIMtmp Sharpnesstmp MAEtmp NMSEtmp FaultyPixeltmp EPRtmp EdgeGenRatiotmp GMSdevtmp MADtmp
        end
        %%
        if NoiseEvaluation
            %%
            % get deep learning segmentation
            phanName = phantom(1:end-5);
            segmDir = [genSegmDir '\' network];
            segmList = dir(segmDir);
            segmList = {segmList(3:end).name};
            liverMask = 1;
            if ~isempty(segmList)
                for m = 1:numel(segmList)
                    genSegmFile = segmList{m};
                    if contains(genSegmFile,phanName) % checks if the phantom name is contained in the selected segmentation
                        liverMask = double(niftiread([segmDir '\' genSegmFile]));
                        liverMask = Nifti_Coordinate_Trafo2(liverMask);
                    end
                end
            end
            if liverMask
                % read in binary masks for liver tissue tissue
%                 liverValue = 51.26;
%                 liverMask = round(XCATData,2) == liverValue; % round due to more precision in actual values than in imshow plot
                liverValue1 = 51.72;
                liverValue2 = 52.39;
                liverValue3 = 53.42;
                liverMask = round(XCATData,2) == liverValue1;
                if numel(nonzeros(liverMask)) == 0
                    liverMask = round(XCATData,2) == liverValue2;
                end
                if numel(nonzeros(liverMask)) == 0
                    liverMask = round(XCATData,2) == liverValue3;
                end
            end
            %%
            % apply masks to CT Image
            liverData = liverMask.*genData; % segment Data using the Liver Mask
            %%
            % calculate 2D NPS
            normFactor = XCATPixelSpacing(1)*XCATPixelSpacing(2); % normalization factor depending on XCAT/gen Pixel spacing
            Data.NPS2D{i} = normFactor*FFT_Segmented_Noise(liverData);
            %%
            % calculate radial NPS, calculate the percentage of inliers and
            % NPS accuracy
            [Data.Frequency{i},Data.NPS{i}] = Radial_from_2D_NPS(Data.NPS2D{i},XCATPixelSpacing(1),1);
            Result.NPSAccuracy{i} = Generated_NPS_Accuracy(Data.NPS{i},Data.ReferenceNPS);
            Data.MPE{i,k} = Generated_NPS_Accuracy(Data.NPS{i},Data.ReferenceNPS);
            %%
            % normalize to 1 and calculate pearson correlation
            tmp = corrcoef(Data.NPS{i}/max(Data.NPS{i}(:)),Data.ReferenceNPS/max(Data.ReferenceNPS(:))); % Pearson correlation of normalized NPS
            Result.NPSCorrelation{i} = tmp(1,2);
            Data.PCC{i,k} = tmp(1,2);
            %%
            % calculate generated noise standard deviation and max ratio
            liverData(liverMask==0) = NaN;
            Result.NoiseMagnitude{i} = nanstd(liverData(:));
            Data.NM{i,k} = nanstd(liverData(:));
            Result.MaxRatio{i} = max(Data.NPS2D{i}(:))/max(Data.NPS{i}(:));
            Data.MMR{i,k} = max(Data.NPS2D{i}(:))/max(Data.NPS{i}(:));
        end  
    end
    %%
    % write the results in a csv
    % write results in csv
    Metrics = cell2table(fieldnames(Result),'VariableNames',{'Metrics'});
    data = struct2cell(Result); % writes the result data in a cell
    data = horzcat(data{:}); % unfolds the result cells in a new cell
    dataOut = cell2table(Cell_End_Results(data),'VariableNames',{network(1:end-10)});
%     dataOut = Cell_End_Results([Result.FSIM,Result.SSIM,Result.MAE,Result.FaultyPixel,Result.EPR,Result.DecVariable,Result.GMSdev]); %,Result.MAD,Result.NMSE]);
    if k == 1
        resultData = [Metrics,dataOut];
        if NoiseEvaluation
            [NPS2DData,NPSData] = deal(cell(nrXCATPhantoms,nrNetworks));
            NPS2DData(:,1) = Data.NPS2D;
            NPSData(:,1) = Data.NPS;
        end
    else
        resultData = [resultData,dataOut];
        if NoiseEvaluation
            NPS2DData(:,k) = Data.NPS2D;
            NPSData(:,k) = Data.NPS;
        end
    end
end
%%
% choose save directory
saveFile = 'A:\Dominik_MIDL\Results\';
writetable(resultData,saveFile)  