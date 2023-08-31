fileName = 'paths\paths_unify.txt';
root = 'Recordings';
FID = fopen(fileName);
data = textscan(FID,'%s');
fclose(FID);
paths = string(data{:})';
dims = size(paths);
% dims(2)
for i = 1:2:dims(2)
    %% load MIDATA.mat for each;
    % unify them, and also unify the
    midata1 = load(strcat(paths(1, i),'/MIData.mat'));
    midata2 = load(strcat(paths(1, i + 1),'/MIData.mat'));
    load(strcat(strcat(paths(1, i),'/EEG_chans.mat')))
    tvec1 = cell2mat(struct2cell(load(strcat(paths(1, i) ,'/trainingVec'))));
    tvec2 = cell2mat(struct2cell(load(strcat(paths(1, i + 1) ,'/trainingVec'))));
    MIData = cat(1, midata1.MIData, midata2.MIData);
    trainingVec = [tvec1 tvec2];
    new_folder = replace(paths(1, i), '001', '\');
    mkdir(new_folder);
    save(strcat(new_folder, 'MIData.mat'), 'MIData');
    save(strcat(new_folder, 'trainingVec.mat'), 'trainingVec');
    save(strcat(new_folder, 'EEG_chans.mat'), 'EEG_chans');
    MI4_featureExtraction(new_folder, 1);
end

fprintf("FINISHED RUN\n");
