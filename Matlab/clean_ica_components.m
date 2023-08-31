function[EEG] = clean_ica_components(EEG, clean_label_min_threshold)
% Inputs:
%   EEG - an EEG struct (including a `data` field, etc.)
%   clean_min_threshold - the value (between 0 and 1) above which 
%                           an IC with unwanted labels will be cleaned from
%                           EEG.data


EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'interrupt','on');
EEG = eeg_checkset( EEG );
EEG = pop_iclabel(EEG, 'default');
EEG = eeg_checkset( EEG );
% The thesholds are for components in this order:
% Brain, Muscle,  Eye, Heart, Line Noise, Channel Noise, Other
EEG = pop_icflag(EEG, ...
    [NaN NaN; ... % Brain
    clean_label_min_threshold 1; ... % Muscle
    clean_label_min_threshold 1; ... % Eye
    clean_label_min_threshold 1; ... % Heart
    NaN NaN; ... % Line noise
    NaN NaN; ... % Channel Noise
    clean_label_min_threshold 1] ... % Other
    );
EEG = eeg_checkset( EEG );

% An empty array as the second argument causes pop_subcomp to remove all
% previously flagged components

% labels are in EEG.etc.ic_classification.ICLabel
EEG = pop_subcomp( EEG, [], 0);
EEG = eeg_checkset( EEG );

end