function [] = Save_Conf_Matrix(c_m,saveDir)
%%saves a plot of the Confusion Matrix
offset = 1 - min(c_m(:));
fig = figure;
imshow(c_m, jet);
h = gca;
h.Visible = 'On';
h.Box = 'off';
colorbar;
h.XTickLabel = compose('%d',(h.XTick - offset));
set(gca,'xaxisLocation','top')
xlabel('Synth')
h.YTickLabel = compose('%d',(h.YTick - offset));
ylabel('XCAT')
savefig(fig,[saveDir 'ConfMatrix'])
% print(fig, 'conf-matrix','-dpng')
end

