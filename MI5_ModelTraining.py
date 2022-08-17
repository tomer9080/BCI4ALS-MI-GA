import scipy.io as sio
import numpy as np
from genetic_selection import GeneticSelectionCV 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import LinearSVC as SVM
from sklearn.naive_bayes import BernoulliNB as NB
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tabulate import tabulate
import sys

# TODO: Add interface for choosing features as we want.
# TODO: How to do the feature mentioned above.
# TODO: Add genetic algorithm feature selection

def cross_validation_on_model(model, k, features, labels):
    kf = KFold(n_splits=k, shuffle=False)

    i = 1
    all_scores = []
    all_models = []
    for train_index, test_index in kf.split(features):
        X_train = features[train_index]
        X_test = features[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
            
        #Train the model
        model.fit(X_train, y_train) #Training the model
        score = accuracy_score(y_test, model.predict(X_test))
        
        all_scores.append(score)
        all_models.append(model)
        i += 1
    
    max_index = np.argmax(all_scores)
    return all_models[max_index]

# TK PC
recordingFolder = "C:\BCI_RECORDINGS\\17-08-22\TK\Sub318324886002"
# recordingFolder = "C:\BCI_RECORDINGS\\09-08-22\RL\Sub316353903001"

# RL PC
# recordingFolder = r'C:\\Users\\Latzres\Desktop\\project\\Recordings\\17-08-22\\TK\Sub318324886002'
# recordingFolder = r'C:\\Users\\Latzres\Desktop\\project\\Recordings\\17-08-22\\RL\Sub316353903002'

# All of the features before train-test partition
all_features = sio.loadmat(recordingFolder + '\AllDataInFeatures.mat')['AllDataInFeatures']
all_labels = sio.loadmat(recordingFolder + '\\trainingVec.mat')['trainingVec'].ravel()
print(all_labels.shape)
#split lables to train and tests
n_samples = all_features.shape[0]  # The total number of samples in the dataset
## Generate a random generator with a fixed seed
rand_gen = np.random.RandomState(0)
## Generating a shuffled vector of indices
indices = np.arange(n_samples)
rand_gen.shuffle(indices)

## Split the indices into 75% train (full) / 25% test
n_samples_train_full = int(n_samples * 0.75)
features_train_ga = all_features[indices[:n_samples_train_full]]
features_test_ga = all_features[indices[n_samples_train_full:]]

labels_train_ga = all_labels[indices[:n_samples_train_full]]
labels_test_ga = all_labels[indices[n_samples_train_full:]]

estimator = SVM(penalty='l2', loss='hinge', multi_class='ovr', C=2, max_iter=30_000)
selector = GeneticSelectionCV(
    estimator,
    cv = 3,
    scoring = "accuracy",
    max_features = 7,
    n_population = 153,
    crossover_proba = 0.5,
    mutation_proba = 0.2,
    n_generations = 60,
    caching = True,
    mutation_independent_proba = 0.025,
    crossover_independent_proba = 0.8
)

selector = selector.fit(features_train_ga, labels_train_ga)

svm_ga_prediction = selector.predict(features_test_ga)
test_results = svm_ga_prediction - labels_test_ga
hit_rate = sum(test_results == 0)/len(labels_test_ga)

svm_ga_row = ['SVM GA', hit_rate, svm_ga_prediction, labels_test_ga, svm_ga_prediction - labels_test_ga]

estimator = LDA()
selector = GeneticSelectionCV(
    estimator,
    cv = 3,
    scoring = "accuracy",
    max_features = 7,
    n_population = 153,
    crossover_proba = 0.5,
    mutation_proba = 0.2,
    n_generations = 60,
    caching = True,
    mutation_independent_proba = 0.025,
    crossover_independent_proba = 0.8
)

selector = selector.fit(features_train_ga, labels_train_ga)

lda_ga_prediction = selector.predict(features_test_ga)
test_results = lda_ga_prediction - labels_test_ga
hit_rate = sum(test_results == 0)/len(labels_test_ga)

lda_ga_row = ['LDA GA', hit_rate, lda_ga_prediction, labels_test_ga, lda_ga_prediction - labels_test_ga]

# features from matlab neighborhood component analysis - takes 10 best features.
features_train = sio.loadmat(recordingFolder + '\FeaturesTrainSelected.mat')['FeaturesTrainSelected']
label_train = sio.loadmat(recordingFolder + '\LabelTrain.mat')['LabelTrain'].ravel()

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

lda_cv_predictor = cross_validation_on_model(LDA(), 9, features_train, label_train)
lda_cv_prediction = lda_cv_predictor.predict(features_test)
test_results = lda_cv_prediction - label_test
hit_rate = sum(test_results == 0)/len(label_test)

lda_cv_row = ['LDA CV', hit_rate, lda_cv_prediction, label_test, lda_cv_prediction - label_test]

#QDA analysis
print("Running QDA analysis...")
qda_predictor = QDA()
qda_predictor.fit(features_train, label_train)
qda_prediction = qda_predictor.predict(features_test)
test_results = qda_prediction - label_test
hit_rate = sum(test_results == 0)/len(label_test)

qda_row = ['QDA', hit_rate, qda_prediction, label_test, qda_prediction - label_test] 


qda_cv_predictor = cross_validation_on_model(QDA(), 9, features_train, label_train)
qda_cv_prediction = qda_cv_predictor.predict(features_test)
test_results = qda_cv_prediction - label_test
hit_rate = sum(test_results == 0)/len(label_test)

qda_cv_row = ['QDA CV', hit_rate, qda_cv_prediction, label_test, qda_cv_prediction - label_test]


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

svm_cv_predictor = cross_validation_on_model(SVM(penalty='l2', loss='hinge', multi_class='ovr', C=2, max_iter=30_000), 9, features_train, label_train)
svm_cv_prediction = svm_cv_predictor.predict(features_test)
test_results = svm_cv_prediction - label_test
hit_rate = sum(test_results == 0)/len(label_test)

svm_cv_row = ['SVM CV', hit_rate, svm_cv_prediction, label_test, svm_cv_prediction - label_test]

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

rf_cv_predictor = cross_validation_on_model(RF(criterion='entropy'), 9, features_train, label_train)
rf_cv_prediction = rf_cv_predictor.predict(features_test)
test_results = rf_cv_prediction - label_test
hit_rate = sum(test_results == 0)/len(label_test)

rf_cv_row = ['RF CV', hit_rate, rf_cv_prediction, label_test, rf_cv_prediction - label_test]

#DT analysis
print("Running DT analysis...")
dt_predictor = DT()
dt_predictor.fit(features_train, label_train)
dt_prediction = dt_predictor.predict(features_test)
test_results = dt_prediction - label_test
hit_rate = sum(test_results == 0)/len(label_test)

dt_row = ['DT', hit_rate, dt_prediction, label_test, dt_prediction - label_test] 

dt_cv_predictor = cross_validation_on_model(DT(), 9, features_train, label_train)
dt_cv_prediction = dt_cv_predictor.predict(features_test)
test_results = dt_cv_prediction - label_test
hit_rate = sum(test_results == 0)/len(label_test)

dt_cv_row = ['DT CV', hit_rate, dt_cv_prediction, label_test, dt_cv_prediction - label_test]

#### ---------- Priniting table ---------- ####
print('')
headers = ["Classifier", "Success Rate", "Classifier Prediction", "Test Labels", "Sub Labels"]
all_rows = [
    lda_row,
    lda_cv_row,
    lda_ga_row,
    qda_row,
    qda_cv_row,
    knn_5_row,
    knn_7_row,
    svm_row,
    svm_cv_row,
    svm_ga_row,
    nb_row,
    rf_row,
    rf_cv_row,
    dt_row,
    dt_cv_row
]
print(tabulate(all_rows, headers=headers))