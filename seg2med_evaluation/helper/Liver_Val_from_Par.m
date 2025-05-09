function [liverValue] = Liver_Val_from_Par(parDir)
% function to read out liver value from XCAT par file
fid1 = fopen(parDir, 'r');
tline = fgetl(fid1);
while ischar(tline)
    tline = fgetl(fid1);
    if contains(char(tline), "liver_activity")
        % Das ist der relevante Leberwert
        liverValue = str2double(erase(tline, "liver_activity = "));
        break
    end
end
end

