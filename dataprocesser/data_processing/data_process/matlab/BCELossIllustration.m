% Create an example image (ground truth)
images1 = rand(256, 256); % Random matrix as an example
images1(images1 > 0.5) = 0.9999;
images1(images1 <= 0.5) = 0.0001;

% Add noise to the image
noise = randn(size(images1)) * 0.01; % Gaussian noise
images2 = images1+noise;

% Ensure values are in the range [0, 1]
% images2 = max(min(images2, 1), 0);

% Plot the images
subplot(1, 2, 1);
imshow(images1);
title('Original Image');

subplot(1, 2, 2);
imshow(images2);
title('Noisy Image');

% Calculate BCEWithLogitsLoss
BCEWithLogitsLoss = calculateBCEWithLogitsLoss(images1, images2);
disp(['BCEWithLogitsLoss: ', num2str(BCEWithLogitsLoss)]);

BCELoss=calculateBCELoss(images1, images2);
disp(['BCELoss: ', num2str(BCELoss)]);

function loss = calculateBCELoss(images1, images2)
    % Convert probabilities to logits
    logits2 = images2;

    % Calculate BCEWithLogitsLoss
    loss = mean(mean(-images1 .* log(logits2) - (1 - images1) .* log(1 - logits2)));
end

function loss = calculateBCEWithLogitsLoss(images1, images2)
    % Convert probabilities to logits
    logits2 = images2;

    % Calculate BCEWithLogitsLoss
    loss = mean(mean(-images1 .* log(sigmoid(logits2)) - (1 - images1) .* log(1 - sigmoid(logits2))));
end

function logit = probToLogit(p)
    % Convert probability to logit
    logit = log(p ./ (1 - p));
end

function s = sigmoid(x)
    % Sigmoid function
    s = 1 ./ (1 + exp(-x));
end
