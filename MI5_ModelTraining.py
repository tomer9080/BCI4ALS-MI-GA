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

# CROSS-VALIDATION
def cross_validation_on_model(model, k, features, labels):
    """
    cross_validation_on_model - given a model, runs a k-fold CV on him, and return a 
    tuple (avg_score, all_scores, all_models)
    :param - model: the model we want to CV
    :param - k: k fold paramater
    :param - features: the features to train the model on
    :param - labels: the label to train the model on
    :return: tuple (avg_score, all_scores, all_models)
    avg_score - the mean score from all folda predictions
    all_scores - all of the scores for each fold
    all_models - all the models that has been trained on the cv session.
    """
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
    
    avg_score = np.average(all_scores)
    print(f"All scores: {all_scores}")
    return avg_score, all_scores, all_models

######### TK PC #########
recordingFolder = "C:\BCI_RECORDINGS\\16-08-22\TK\Sub318324886001"
recordingFolder_2 = "C:\BCI_RECORDINGS\\16-08-22\TK\Sub318324886002"
# recordingFolder = "C:\BCI_RECORDINGS\\16-08-22\RL\Sub316353903002"

######### RL PC #########
# recordingFolder = r'C:\\Users\\Latzres\Desktop\\project\\Recordings\\16-08-22\\TK\Sub318324886002'
# recordingFolder = r'C:\\Users\\Latzres\Desktop\\project\\Recordings\\16-08-22\\RL\Sub316353903002'

# All of the features before train-test partition
all_features = sio.loadmat(recordingFolder + '\AllDataInFeatures.mat')['AllDataInFeatures']
all_labels = sio.loadmat(recordingFolder + '\\trainingVec.mat')['trainingVec'].ravel()
test_indices = sio.loadmat(recordingFolder + '\\testIdx.mat')['testIdx'].ravel()
nca_selected_idx = sio.loadmat(recordingFolder + '\\SelectedIdx.mat')['SelectedIdx'].ravel() - 1 
if sys.argv[1] == '2':
    all_features_2 = sio.loadmat(recordingFolder_2 + '\AllDataInFeatures.mat')['AllDataInFeatures']
    all_labels_2 = sio.loadmat(recordingFolder_2 + '\\trainingVec.mat')['trainingVec'].ravel()
    test_indices_2 = sio.loadmat(recordingFolder_2 + '\\testIdx.mat')['testIdx'].ravel()
    nca_selected_idx_2 = sio.loadmat(recordingFolder_2 + '\\SelectedIdx.mat')['SelectedIdx'].ravel() - 1
    
    all_features = np.concatenate((all_features, all_features_2), axis=0)
    all_labels = np.concatenate((all_labels, all_labels_2), axis=0)
    test_indices = np.concatenate((test_indices, test_indices_2), axis=0)
    nca_selected_idx = np.concatenate((nca_selected_idx[:5], nca_selected_idx_2[:5]), axis=0)

print(all_features.shape, all_labels.shape, test_indices.shape, nca_selected_idx.shape, all_features[:,nca_selected_idx].shape)
test_indices = test_indices - 1
train_indices = [i for i in range(len(all_labels)) if i not in test_indices]


# GENETIC ALGORITHEM analysis
features_train_ga = all_features[train_indices]
features_test_ga = all_features[test_indices]

labels_train_ga = all_labels[train_indices]
labels_test_ga = all_labels[test_indices]

estimator = SVM(penalty='l2', loss='hinge', multi_class='ovr', C=0.1, max_iter=30_000)
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

if sys.argv[1] == '2':
    features_train_2 = sio.loadmat(recordingFolder_2 + '\FeaturesTrainSelected.mat')['FeaturesTrainSelected']
    label_train_2 = sio.loadmat(recordingFolder_2 + '\LabelTrain.mat')['LabelTrain'].ravel()

    features_test_2 = sio.loadmat(recordingFolder_2 + '\FeaturesTest.mat')['FeaturesTest']
    label_test_2 = sio.loadmat(recordingFolder_2 + '\LabelTest.mat')['LabelTest'].ravel()
    
    features_train = np.concatenate((features_train, features_train_2), axis=0)
    label_train = np.concatenate((label_train, label_train_2), axis=0)
    features_test = np.concatenate((features_test, features_test_2), axis=0)
    label_test = np.concatenate((label_test, label_test_2), axis=0)



#LDA analysis
print("Running LDA analysis...")
lda_predictor = LDA()
lda_predictor.fit(features_train, label_train)
lda_prediction = lda_predictor.predict(features_test)
test_results = lda_prediction - label_test
hit_rate = sum(test_results == 0)/len(label_test)

lda_row = ['LDA', hit_rate, lda_prediction, label_test, lda_prediction - label_test] 


lda_cv_predictor = cross_validation_on_model(LDA(), 5, all_features[:,nca_selected_idx], all_labels) 
hit_rate = lda_cv_predictor[0]

lda_cv_row = ['LDA CV', hit_rate, [], label_test, []]

#QDA analysis
print("Running QDA analysis...")
qda_predictor = QDA()
qda_predictor.fit(features_train, label_train)
qda_prediction = qda_predictor.predict(features_test)
test_results = qda_prediction - label_test
hit_rate = sum(test_results == 0)/len(label_test)

qda_row = ['QDA', hit_rate, qda_prediction, label_test, qda_prediction - label_test] 


qda_cv_predictor = cross_validation_on_model(QDA(), 5, all_features[:,nca_selected_idx], all_labels)
hit_rate = qda_cv_predictor[0]

qda_cv_row = ['QDA CV', hit_rate, [], label_test, []]


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

svm_cv_predictor = cross_validation_on_model(SVM(penalty='l2', loss='hinge', multi_class='ovr', C=2, max_iter=30_000), 5, all_features[:,nca_selected_idx], all_labels)
hit_rate = svm_cv_predictor[0]

svm_cv_row = ['SVM CV', hit_rate, [], label_test, []]

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

rf_cv_predictor = cross_validation_on_model(RF(criterion='entropy'), 5, all_features[:,nca_selected_idx], all_labels)
hit_rate = rf_cv_predictor[0]

rf_cv_row = ['RF CV', hit_rate, [], label_test, []]

#DT analysis
print("Running DT analysis...")
dt_predictor = DT()
dt_predictor.fit(features_train, label_train)
dt_prediction = dt_predictor.predict(features_test)
test_results = dt_prediction - label_test
hit_rate = sum(test_results == 0)/len(label_test)

dt_row = ['DT', hit_rate, dt_prediction, label_test, dt_prediction - label_test] 

dt_cv_predictor = cross_validation_on_model(DT(), 5, all_features[:,nca_selected_idx], all_labels)
hit_rate = dt_cv_predictor[0]

dt_cv_row = ['DT CV', hit_rate, [], label_test, []]

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