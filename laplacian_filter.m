function [EEG] = laplacian_filter(EEG,main_channel,surrounding_electrodes)
%laplacian_filter - The function gets EEG object, main electrode channel,
% surrounding electrodes channels and substracts the surrounding electrodes'
%   mean from the main electrode data.
f2 = mean(EEG.data(surrounding_electrodes,:));
EEG.data(main_channel,:) = EEG.data(main_channel,:) - f2;
end