function [test_results] = MI5_modelTraining(recordingFolder)
% MI5_LearnModel_Scaffolding outputs a weight vector for all the features
% using a simple multi-class linear approach.
% Add your own classifier (SVM, CSP, DL, CONV, Riemann...), and make sure
% to add an accuracy test.

%% This code is part of the BCI-4-ALS Course written by Asaf Harel
% (harelasa@post.bgu.ac.il) in 2021. You are free to use, change, adapt and
% so on - but please cite properly if published.
%% Read the features & labels 

recordingFolder ='C:\Users\Latzres\Desktop\project\Recordings\10-08-22\TK\Sub318324886';


FeaturesTrain = cell2mat(struct2cell(load(strcat(recordingFolder,'\FeaturesTrainSelected.mat'))));   % features for train set
LabelTrain = cell2mat(struct2cell(load(strcat(recordingFolder,'\LabelTrain'))));                % label vector for train set

% label vector
LabelTest = cell2mat(struct2cell(load(strcat(recordingFolder,'\LabelTest'))));      % label vector for test set
load(strcat(recordingFolder,'\FeaturesTest.mat'));                                  % features for test set
% FeaturesTest1 = load(strcat(recordingFolder,'\FeaturesTest.mat')).FeaturesTest;                                  % features for test set


%FeaturesTrain = cell2mat(struct2cell(load(strcat(recordingFolder,'/FeaturesTrain.mat'))));   % features for train set
%LabelTrain = cell2mat(struct2cell(load(strcat(recordingFolder,'/LabelTrain'))));                % label vector for train set

% label vector
%LabelTest = cell2mat(struct2cell(load(strcat(recordingFolder,'/LabelTest'))));      % label vector for test set
%FeaturesTest = load(strcat(recordingFolder,'/FeaturesTest.mat'));                                  % features for test set

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% Split to train and validation sets %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% test data
testPrediction = classify(FeaturesTest,FeaturesTrain,LabelTrain,'linear');          % classify the test set using a linear classification object (built-in Matlab functionality)
W = LDA(FeaturesTrain,LabelTrain);                                                  % train a linear discriminant analysis weight vector (first column is the constants)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% Add your own classifier %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SVM ECOC %%%%%%%%%%%%%%%%%%%%%%%%%%%%
SVMModel = fitcecoc(FeaturesTrain,LabelTrain);

svm_labels = predict(SVMModel,FeaturesTest);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
KNNModel = fitcknn(FeaturesTrain,LabelTrain,'NumNeighbors',5);
KNNPred = predict(KNNModel,FeaturesTest);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NBModel = fitcnb(FeaturesTrain, LabelTrain);
NBPred = predict(NBModel, FeaturesTest);

%% Test data
% test prediction from linear classifier
test_results = (testPrediction'-LabelTest);                                         % prediction - true labels = accuracy
test_results = (sum(test_results == 0)/length(LabelTest))*100;
disp(['test accuracy - ' num2str(test_results) '%'])

test_results = (svm_labels'-LabelTest);                                         % prediction - true labels = accuracy
test_results = (sum(test_results == 0)/length(LabelTest))*100;
disp(['SVM test accuracy - ' num2str(test_results) '%'])


test_results_knn = (KNNPred'-LabelTest);                                         % prediction - true labels = accuracy
test_results_knn = (sum(test_results_knn == 0)/length(LabelTest))*100;
disp(['KNN test accuracy - ' num2str(test_results_knn) '%'])

test_results_nb = (NBPred'-LabelTest);                                         % prediction - true labels = accuracy
test_results_nb = (sum(test_results_nb == 0)/length(LabelTest))*100;
disp(['NB test accuracy - ' num2str(test_results_nb) '%'])

save(strcat(recordingFolder,'\TestResults.mat'),'test_results');                    % save the accuracy results
save(strcat(recordingFolder,'\WeightVector.mat'),'W');                              % save the model (W)

end


