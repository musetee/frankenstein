function [histCC,synthHist] = Compare_Save_Histograms(pat,synth,edges,saveDir,phan)
% Generate Histograms
nrVoxels = size(synth,1)*size(synth,2)*size(synth,3);
synthHist = histcounts(synth,edges)/nrVoxels;
ctrs = edges(1)+(1:length(edges)-1).*diff(edges);   % Create Centres
fig = figure;

b = histogram('BinCounts', pat', 'BinEdges', edges,'facecolor',[1 0 0],'facealpha',.5,'edgecolor','none');
hold on
b2 = histogram('BinCounts', synthHist', 'BinEdges', edges,'facecolor',[0 0 1],'facealpha',.5,'edgecolor','none');

% b = bar(ctrs, [pat; synthHist]',0.9,'hist');
% b.FaceColor = [0 0 1];
% b.EdgeColor = [0 0 1];
% b2.EdgeColor = [0.9290, 0.6940, 0.1250];
% b2.FaceColor = [0.9290, 0.6940, 0.1250];
xlim([min(edges(:)) max(edges(:))])
legend('Patient','Synthetic')
% print(fig, 'hist','-dpng')
savefig(fig,[saveDir phan '.fig'])

thres=0.0001
pat2 = pat(pat>=thres & synthHist>=thres);
synthHist2 = synthHist(synthHist>=thres & pat>=thres);

% pat3 = pat2(synthHist2>=0.001);
% synthHist3 = synthHist2(pat2>=0.001);

tmp = corrcoef(pat2,synthHist2); % Pearson correlation
histCC = tmp(1,2);
end