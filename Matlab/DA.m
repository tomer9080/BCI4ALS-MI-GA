fileName = 'paths\paths_unify.txt';
root = 'Recordings';
FID = fopen(fileName);
data = textscan(FID,'%s');
fclose(FID);
paths = string(data{:})';
dims = size(paths);
for i = 1:dims(2)
    load(strcat(paths(1, i), '/MIData.mat'));
    load(strcat(paths(1, i), '/AllDataInFeatures.mat'));
    midata_dim = size(MIData);
    MICov = ones(midata_dim(1), midata_dim(2), midata_dim(2));
    MIFFTCov = ones(midata_dim(1), midata_dim(2)*2, midata_dim(2)*2);
    MIFEATURESCov = ones(60, 245, 245);
    for j = 1:midata_dim(1)
        trial = MIData(j,:,:);
        trial_s = squeeze(trial);
        
        % Time series Cov
        tmp_cov = cov(trial_s');
        MICov(j,:,:) = tmp_cov;
        
        % Time + FFT series Cov
        trial_fft = abs(fftshift(fft(trial_s')));
        trial_unified = cat(1, trial_s, trial_fft');
        tmp_cov = cov(trial_unified');
        MIFFTCov(j,:,:) = tmp_cov;
    end 
    
    for j = 1:60
        trial = AllDataInFeatures(j,:);
        trial_s = squeeze(trial);
        
        % Features Cov
        tmp_cov = cov(trial_s');
        MIFEATURESCov(j,:,:) = abs(tmp_cov);
    end
    save(strcat(paths(1, i), '\MICov.mat'), 'MICov');
    save(strcat(paths(1, i), '\MIFFTCov.mat'), 'MIFFTCov');
    save(strcat(paths(1, i), '\MIFEATURESCov.mat'), 'MIFEATURESCov');
        
end

DA_OT(paths(1,1),paths(1,2));
