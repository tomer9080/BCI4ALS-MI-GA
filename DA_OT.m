function [] = DA_OT(folder1, folder2)

    micov1 = load(strcat(folder1, '/MICov.mat')).MICov;
    micov2 = load(strcat(folder2, '/MICov.mat')).MICov;
    
    
    tvec1 = cell2mat(struct2cell(load(strcat(folder1 ,'/trainingVec'))));
    tvec2 = cell2mat(struct2cell(load(strcat(folder2 ,'/trainingVec'))));
    
    vS     = [1 * ones(1, length(tvec1)), 2 * ones(1, length(tvec2))];
    tvec = [tvec1, tvec2];

    %% No OT
    Covs_tmp = cat(1, micov1, micov2);
    for i = 1:size(Covs_tmp, 1)
        Covs{i} = squeeze(Covs_tmp(i,:,:));
    end
    mX     = CovsToVecs(cat(3, Covs{:}));

    %%
    mTSNE = tsne(mX')';
    figure; PlotData(mTSNE, tvec, vS);
    subplot(1,2,1); title('Before applying OT', 'Interpreter', 'Latex');

    %% Applying OT
    N1      = length(micov1);
    N2      = length(micov2);
%     vP1     = ones(N1, 1) / N1;
%     vP2     = ones(N1, 1) / N2;
%     mC      = PRdist2(micov1, micov2).^2;
    mPlan   = SinkhornRegOptimalTransport(to_braced(micov1), to_braced(micov2), tvec1); %-- Supervised
    % mPlan   = SinkhornOptimalTransport(mC, vP1, vP2); %-- Unsupervised
    Covs1OT = ApplyPlan(to_braced(micov1), to_braced(micov2), mPlan, 20);
    

    %% 
    Covs = [Covs1OT, to_braced(micov2)];
    mX     = CovsToVecs(cat(3, Covs{:}));

    %%  
    mX1 = mX(:, vS == 1);
    mX2 = mX(:, vS == 2);
    linaerSvmTemplate = templateSVM('Standardize', false);
    mdlLinearSVM      = fitcecoc(mX1', tvec1, 'Learners', linaerSvmTemplate);
    res               = mean( mdlLinearSVM.predict(mX2') == tvec2' );

    %%
    mTSNE = tsne(mX')';
    figure; PlotData(mTSNE, tvec, vS);
    subplot(1,2,1); title('After applying OT', 'Interpreter', 'Latex');
    subplot(1,2,2); title(['Accuracy - $', num2str(100*res), '\%$'], 'Interpreter', 'Latex');


    %%
    function PlotData(mX, vClass, vS)

        vMarker = 'od';
        vColorS = 'br';
        vColorC = 'gmyk';

        vUniqueS  = unique(vS);

        %--
        subplot(1,2,2);
        for cc = 1 : 3
            vIdxC = vClass == cc;
            for ss = 1 : 2
                marker = vMarker(ss);
                vIdxS  = vS == vUniqueS(ss);
                color  = vColorS(ss);
                vIdx   = vIdxS & vIdxC;
                scatter(mX(1,vIdx), mX(2,vIdx), 50, color, marker, 'Fill', 'MarkerEdgeColor', 'k'); hold on;
            end
        end

        h = legend({['Subject - ', num2str(vUniqueS(1))];
                    ['Subject - ', num2str(vUniqueS(2))]}, ...
                    'FontSize', 12, 'Location', 'Best'); set(h, 'Color', 'None');
        axis tight;

        %--
        subplot(1,2,1);
        for ss = 1 : 2
            marker = vMarker(ss);
            vIdxS  = vS == vUniqueS(ss);
            for cc = 1 : 4
                color  = vColorC(cc);
                vIdxC = vClass == cc;
                vIdx  = vIdxS & vIdxC;
                scatter(mX(1,vIdx), mX(2,vIdx), 50, color, marker, 'Fill', 'MarkerEdgeColor', 'k'); hold on;
            end
        end

        h = legend({'Left Hand', 'Right Hand', 'Foot', 'Tongue'}, 'FontSize', 12, 'Location', 'Best'); set(h, 'Color', 'None');
        axis tight;
    end

    function braced_vec = to_braced(vectorini)
       for a = 1:size(vectorini, 1)
        braced_vec{a} = squeeze(vectorini(a,:,:));
       end 
    end

end