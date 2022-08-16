import scipy.io as sio
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import LinearSVC as SVM
from sklearn.naive_bayes import BernoulliNB as NB
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.tree import DecisionTreeClassifier as DT
from tabulate import tabulate
import sys


# TODO: Add interface for choosing features as we want.
# TODO: How to do the feature mentioned above.


# recordingFolder = "C:\BCI_RECORDINGS\\09-08-22\TK\Sub318324886002"
recordingFolder = r'C:\\Users\\Latzres\Desktop\\project\\Recordings\\16-08-22\\TK\Sub318324886002'
# recordingFolder = r'C:\\Users\\Latzres\Desktop\\project\\Recordings\\16-08-22\\RL\Sub316353903002'
# recordingFolder = "C:\BCI_RECORDINGS\\09-08-22\RL\Sub316353903001"
# recordingFolder = r'C:\\Users\\Latzres\\Desktop\\project\\Recordings\\14-08-22\\Sub20220814002'


features_train = sio.loadmat(recordingFolder + '\FeaturesTrainSelected.mat')['FeaturesTrainSelected']
features_train_all = sio.loadmat(recordingFolder + '\FeaturesTrain.mat')['FeaturesTrain']
print(features_train_all.shape)
label_train = sio.loadmat(recordingFolder + '\LabelTrain.mat')['LabelTrain'].ravel()

features = sys.argv[1:] # taking the features from the command line
# Attaching number to each feature name that comes from cmdline.
features_hash = {} 

# Label Vector
features_test = sio.loadmat(recordingFolder + '\FeaturesTest.mat')['FeaturesTest'] 
label_test = sio.loadmat(recordingFolder + '\LabelTest.mat')['LabelTest'].ravel()


#LDA analysis
print("Running LDA analysis...")
lda_predictor = LDA()
lda_predictor.fit(features_train, label_train)
lda_prediction = lda_predictor.predict(features_test)
test_results = lda_prediction - label_test
hit_rate = sum(test_results == 0)/len(label_test)

lda_row = ['LDA', hit_rate, lda_prediction, label_test, lda_prediction - label_test] 


#QDA analysis
print("Running QDA analysis...")
qda_predictor = QDA()
qda_predictor.fit(features_train, label_train)
qda_prediction = qda_predictor.predict(features_test)
test_results = qda_prediction - label_test
hit_rate = sum(test_results == 0)/len(label_test)

qda_row = ['QDA', hit_rate, qda_prediction, label_test, qda_prediction - label_test] 


#KNN analysis

#KNN-5
print("Running KNN-5 analysis...")
knn_predictor = KNN(5)
knn_predictor.fit(features_train, label_train)
knn_prediction = knn_predictor.predict(features_test)
test_results = knn_prediction - label_test
hit_rate = sum(test_results == 0)/len(label_test)

knn_5_row = ['KNN-5', hit_rate, knn_prediction, label_test, knn_prediction - label_test] 

#KNN-7
print("Running KNN-7 analysis...")
knn_predictor = KNN(7)
knn_predictor.fit(features_train, label_train)
knn_prediction = knn_predictor.predict(features_test)
test_results = knn_prediction - label_test
hit_rate = sum(test_results == 0)/len(label_test)

knn_7_row = ['KNN-7', hit_rate, knn_prediction, label_test, knn_prediction - label_test] 

#SVM analysis
# Need to check convergence problem.
# Need to play with params.
print("Running SVM analysis...")
svm_predictor = SVM(penalty='l2', loss='hinge', multi_class='ovr', C=2, max_iter=30_000)
svm_predictor.fit(features_train, label_train)
svm_prediction = svm_predictor.predict(features_test)
test_results = svm_prediction - label_test
hit_rate = sum(test_results == 0)/len(label_test)

svm_row = ['SVM', hit_rate, svm_prediction, label_test, svm_prediction - label_test] 

#NB analysis
print("Running NB analysis...")
nb_predictor = NB()
nb_predictor.fit(features_train, label_train)
nb_prediction = nb_predictor.predict(features_test)
test_results = nb_prediction - label_test
hit_rate = sum(test_results == 0)/len(label_test)

nb_row = ['NB', hit_rate, nb_prediction, label_test, nb_prediction - label_test] 

#RF analysis
print("Running RF analysis...")
rf_predictor = RF(criterion='entropy')
rf_predictor.fit(features_train, label_train)
rf_prediction = rf_predictor.predict(features_test)
test_results = rf_prediction - label_test
hit_rate = sum(test_results == 0)/len(label_test)

rf_row = ['RF', hit_rate, rf_prediction, label_test, rf_prediction - label_test] 

#DT analysis
print("Running DT analysis...")
dt_predictor = DT()
dt_predictor.fit(features_train, label_train)
dt_prediction = dt_predictor.predict(features_test)
test_results = dt_prediction - label_test
hit_rate = sum(test_results == 0)/len(label_test)

dt_row = ['DT', hit_rate, dt_prediction, label_test, dt_prediction - label_test] 

#### ---------- Priniting table ---------- ####
print('')
headers = ["Classifier", "Success Rate", "Classifier Prediction", "Test Labels", "Sub Labels"]
all_rows = [
    lda_row,
    qda_row,
    knn_5_row,
    knn_7_row,
    svm_row,
    nb_row,
    rf_row,
    dt_row
]
print(tabulate(all_rows, headers=headers))