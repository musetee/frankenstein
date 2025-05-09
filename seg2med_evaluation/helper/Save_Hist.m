function [] = Save_Hist(pat,synthHist,edges,saveDir)
%%saves a plot of the Confusion Matrix
ctrs = edges(1)+(1:length(edges)-1).*diff(edges);   % Create Centres
fig = figure;
% b = bar(ctrs, [pat; synthHist]',0.9,'hist');
b = histogram('BinCounts', pat', 'BinEdges', edges,'facecolor',[1 0 0],'facealpha',.5,'edgecolor','none');
hold on
b2 = histogram('BinCounts', synthHist', 'BinEdges', edges,'facecolor',[0 0 1],'facealpha',.5,'edgecolor','none');
% b(1).FaceColor = [0 0 1];
% b(1).EdgeColor = [0 0 1];
% b(2).EdgeColor = [0.9290, 0.6940, 0.1250];
% b(2).FaceColor = [0.9290, 0.6940, 0.1250];
xlim([min(edges(:)) max(edges(:))])
legend('Patient','Synthetic')
% print(fig, 'hist','-dpng')
savefig(fig,[saveDir 'HistogramComparison'])
end

