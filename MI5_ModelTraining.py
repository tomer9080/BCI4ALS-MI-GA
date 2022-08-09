import scipy.io as sio
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import LinearSVC as SVM
from sklearn.naive_bayes import BernoulliNB as NB

recordingFolder = "C:\BCI_RECORDINGS\\09-08-22\TK\Sub318324886002"
# recordingFolder = "C:\BCI_RECORDINGS\\09-08-22\RL\Sub316353903001"

features_train = sio.loadmat(recordingFolder + '\FeaturesTrainSelected.mat')['FeaturesTrainSelected']
label_train = sio.loadmat(recordingFolder + '\LabelTrain.mat')['LabelTrain'].ravel()

# Label Vector
features_test = sio.loadmat(recordingFolder + '\FeaturesTest.mat')['FeaturesTest'] 
label_test = sio.loadmat(recordingFolder + '\LabelTest.mat')['LabelTest'].ravel()

print(features_train.shape, label_train.shape, features_test.shape, label_test.shape)

lda_predictor = LDA()
lda_predictor.fit(features_train, label_train)
lda_prediction = lda_predictor.predict(features_test)
print(lda_prediction)
print(label_test)
test_results = lda_prediction - label_test
print(sum(test_results == 0)/len(label_test))

qda_predictor = QDA()
qda_predictor.fit(features_train, label_train)
qda_prediction = qda_predictor.predict(features_test)
print(qda_prediction)
print(label_test)
test_results = qda_prediction - label_test
print(sum(test_results == 0)/len(label_test))

knn_predictor = KNN(5)
knn_predictor.fit(features_train, label_train)
knn_prediction = knn_predictor.predict(features_test)
print(knn_prediction)
print(label_test)
test_results = knn_prediction - label_test
print(sum(test_results == 0)/len(label_test))

knn_predictor = KNN(7)
knn_predictor.fit(features_train, label_train)
knn_prediction = knn_predictor.predict(features_test)
print(knn_prediction)
print(label_test)
test_results = knn_prediction - label_test
print(sum(test_results == 0)/len(label_test))

# Need to check convergence problem.
svm_predictor = SVM(penalty='l2', loss='hinge', multi_class='ovr', C=2, max_iter=30_000)
svm_predictor.fit(features_train, label_train)
svm_prediction = svm_predictor.predict(features_test)
print(svm_prediction)
print(label_test)
test_results = svm_prediction - label_test
print(sum(test_results == 0)/len(label_test))

nb_predictor = NB()
nb_predictor.fit(features_train, label_train)
nb_prediction = nb_predictor.predict(features_test)
print(nb_prediction)
print(label_test)
test_results = nb_prediction - label_test
print(sum(test_results == 0)/len(label_test))
