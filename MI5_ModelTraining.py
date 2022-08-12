import scipy.io as sio
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import LinearSVC as SVM
from sklearn.naive_bayes import BernoulliNB as NB

# recordingFolder = "C:\BCI_RECORDINGS\\10-08-22\TK\Sub318324886"
recordingFolder = r'C:\\Users\\Latzres\Desktop\\project\\Recordings\\10-08-22\\TK\Sub318324886002'

# recordingFolder = "C:\BCI_RECORDINGS\\09-08-22\RL\Sub316353903001"

features_train = sio.loadmat(recordingFolder + '\FeaturesTrainSelected.mat')['FeaturesTrainSelected']
label_train = sio.loadmat(recordingFolder + '\LabelTrain.mat')['LabelTrain'].ravel()

# Label Vector
features_test = sio.loadmat(recordingFolder + '\FeaturesTest.mat')['FeaturesTest'] 
label_test = sio.loadmat(recordingFolder + '\LabelTest.mat')['LabelTest'].ravel()

print(features_train.shape, label_train.shape, features_test.shape, label_test.shape)


#LDA analysis
print("starting LDA analysis: \n")
lda_predictor = LDA()
lda_predictor.fit(features_train, label_train)
lda_prediction = lda_predictor.predict(features_test)

print("LDA predction: \n")
print(lda_prediction)
print("LDA label test: \n")
print(label_test)

test_results = lda_prediction - label_test
print("test results: \n")
print(sum(test_results == 0)/len(label_test))

#QDA analysis
print("starting QDA analysis: \n")
qda_predictor = QDA()
qda_predictor.fit(features_train, label_train)
qda_prediction = qda_predictor.predict(features_test)

print("QDA predction: \n")
print(qda_prediction)
print("QDA label test: \n")
print(label_test)

test_results = qda_prediction - label_test
print("test results: \n")
print(sum(test_results == 0)/len(label_test))

#KNN analysis

#KNN-5
print("starting KNN-5 analysis: \n")
knn_predictor = KNN(5)
knn_predictor.fit(features_train, label_train)
knn_prediction = knn_predictor.predict(features_test)

print("KNN-5 predction: \n")
print(knn_prediction)
print("KNN-5 label test: \n")
print(label_test)

test_results = knn_prediction - label_test
print("test results: \n")
print(sum(test_results == 0)/len(label_test))

#KNN-7
print("starting KNN-7 analysis: \n")
knn_predictor = KNN(7)
knn_predictor.fit(features_train, label_train)
knn_prediction = knn_predictor.predict(features_test)

print("KNN-7 predction: \n")
print(knn_prediction)
print("KNN-7 label test: \n")
print(label_test)

test_results = knn_prediction - label_test
print("test results: \n")
print(sum(test_results == 0)/len(label_test))


#SVM analysis
# Need to check convergence problem.
print("starting SVM analysis: \n")
svm_predictor = SVM(penalty='l2', loss='hinge', multi_class='ovr', C=2, max_iter=30_000)
svm_predictor.fit(features_train, label_train)
svm_prediction = svm_predictor.predict(features_test)

print("SVM predction: \n")
print(svm_prediction)
print("SVM label test: \n")
print(label_test)

test_results = svm_prediction - label_test
print("test results: \n")
print(sum(test_results == 0)/len(label_test))


#NB analysis
print("starting NB analysis: \n")
nb_predictor = NB()
nb_predictor.fit(features_train, label_train)
nb_prediction = nb_predictor.predict(features_test)

print("NB predction: \n")
print(nb_prediction)
print("NB label test: \n")
print(label_test)

test_results = nb_prediction - label_test
print("test results: \n")
print(sum(test_results == 0)/len(label_test))
