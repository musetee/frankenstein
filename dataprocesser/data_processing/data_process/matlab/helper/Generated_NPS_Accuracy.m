function [accuracy] = Generated_NPS_Accuracy(genNPS,refNPS)
% calculates the average percent amplitude difference of a generated NPS
% from a reference NPS, provided they share the same frequency axis,
% calculates the NPS accuracy according to Dolly 2016
accuracy = mean(abs(genNPS - refNPS)/max(refNPS(:))*100);


