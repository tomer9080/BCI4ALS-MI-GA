%% Parameters (this needs to change according to your system):
addpath 'C:\BCIToolBox\eeglab2021.1'
recordingFolder = 'C:\BCI_RECORDINGS\23-05-22\TK1\Sub318324886';
highLim = 40;                               % filter data under 40 Hz
lowLim = 0.5;                               % filter data above 0.5 Hz
recordingFile = strcat(recordingFolder,'\EEG.XDF');

% (1) Load subject data (assume XDF)
EEG = pop_loadxdf(recordingFile, 'streamtype', 'EEG', 'exclude_markerstreams', {});
EEG.setname = 'MI_sub';

% (2) Plot raw data
timevltg(EEG);
figure('name', 'PSD of raw data');
psdplot(EEG);

% (3) Update channel names
EEG_chans(1,:) = 'C03';
EEG_chans(2,:) = 'C04';
EEG_chans(3,:) = 'C0Z';
EEG_chans(4,:) = 'FC1';
EEG_chans(5,:) = 'FC2';
EEG_chans(6,:) = 'FC5';
EEG_chans(7,:) = 'FC6';
EEG_chans(8,:) = 'CP1';
EEG_chans(9,:) = 'CP2';
EEG_chans(10,:) = 'CP5';
EEG_chans(11,:) = 'CP6';
EEG_chans(12,:) = 'O01';
EEG_chans(13,:) = 'O02';
EEG_chans(14,:) = 'NA1';
EEG_chans(15,:) = 'NA2';
EEG_chans(16,:) = 'NA3';

%% Filters
% (4) Low-pass filter
EEG = pop_eegfiltnew(EEG, 'hicutoff',highLim,'plotfreqz',1);    % remove data above
figure('name', 'PSD after low-pass filter');
psdplot(EEG);

% (5) High-pass filter
EEG = pop_eegfiltnew(EEG, 'locutoff',lowLim,'plotfreqz',1);     % remove data under
figure('name', 'PSD after low-pass & high-pass filter');
psdplot(EEG);

% (6) LaPlacian filter for C03. Electrodes channels are hard coded (ex C03=1), make
% sure to update if there is any change
EEG = laplacian_filter(EEG, 1, [4,6,8,10]);

% (7) LaPlacian filter for C04. Electrodes channels are hard coded (ex C04=2), make
% sure to update if there is any change
EEG = laplacian_filter(EEG, 2, [5,7,9,11]);

% (8) Plot comparison
timevltg(EEG);