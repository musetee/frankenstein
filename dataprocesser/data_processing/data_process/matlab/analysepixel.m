img1 = imread('14_0_fake.jpg');
img2 = imread('14_0_ct.jpg');
%img1=minmaxnorm(img1);
%img2 = zscore(double(img2));
% Define the figure's position and size
left = 100;     % X-coordinate of the left edge of the figure
bottom = 100;   % Y-coordinate of the bottom edge of the figure
width = 1000;    % Width of the figure in pixels
height = 400;   % Height of the figure in pixels

% Set the figure's position and size
fig1=figure();
set(fig1, 'Position', [left, bottom, width, height]);
subplot(1, 2, 1);
imshow(img1)
title('DDPM');
subplot(1, 2, 2);
imshow(img2)
title('CT');

fig2=figure();
set(fig2, 'Position', [left, bottom, width, height]);
% Plot the first image's pixel value distribution in the first subfigure
subplot(1, 2, 1);
[counts1, binValues1] = imhist(img1);
bar(binValues1, counts1, 'b', 'DisplayName', 'DDPM');
title('DDPM - Pixel Value Distribution');
xlabel('Pixel Value');
ylabel('Frequency');

% Plot the second image's pixel value distribution in the second subfigure
subplot(1, 2, 2);
[counts2, binValues2] = imhist(img2);
bar(binValues2, counts2, 'r', 'DisplayName', 'CT');
title('CT - Pixel Value Distribution');
xlabel('Pixel Value');
ylabel('Frequency');
