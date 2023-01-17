fileName = 'paths\paths_unified.txt';
root = 'Recordings';
FID = fopen(fileName);
data = textscan(FID,'%s');
fclose(FID);
paths = string(data{:})';
dims = size(paths);
for i = 1:dims(2)
    load(strcat(paths(1, i), '/MIData.mat'));
    midata_dim = size(MIData);
    MICov = ones(midata_dim(1), midata_dim(2), midata_dim(2));
    for j = 1:midata_dim(1)
        trial = MIData(j,:,:);
        trial_dims = size(trial);
        trial_s = squeeze(trial);
        tmp_cov = cov(trial_s');
        MICov(j,:,:) = tmp_cov;
    end 
    save(strcat(paths(1, i), '\MICov.mat'), 'MICov');
end

DA_OT(paths(1,1),paths(1,3));
