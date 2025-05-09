function normalized_data = minmaxnorm(data)
% Define the desired min and max values
desired_min = -1;
desired_max = 1;

% Calculate the min and max values of your data
min_value = min(data);
max_value = max(data);

% Perform min-max normalization
normalized_data = ((data - min_value) ./ (max_value - min_value)) .* (desired_max - desired_min) + desired_min;
