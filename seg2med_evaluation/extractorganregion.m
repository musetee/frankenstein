function maskData = extractorganregion(maskData, organ_value)
% Set all values in maskData that are not equal to organ_value to 0
    maskData(maskData ~= organ_value) = 0;
    
    % Optionally, set the specified organ region to a binary value (e.g., 1) for a binary mask
    maskData(maskData == organ_value) = 1;

end

