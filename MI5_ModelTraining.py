import scipy.io as sio
import numpy as np
from genetic_selection import GeneticSelectionCV 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
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

features_names_list = ['BP_ALPHA', 'BP_BETA', 'BP_GAMMA', 'BP_DELTA', 'BP_THETA', 'RTP', 'SPEC_MOM', 'SPEC_EDGE', 'SPEC_ENT', 'SLOPE', 'INTERCEPT', 'MEAN_FREQ', 'OCC_BAND', 'POWER_BAND', 'WLT_ENT', 'KURT', 'SKEW', 'VAR', 'STD', 'LOG_ENE_ENT']
headers = np.array(['CSP1', 'CSP2', 'CSP3'] + [f'E{i}_{feature}' for i in range(1,12) for feature in features_names_list])

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


def classify_results(model, model_name, features_train, label_train, features_test, label_test, nca_idx, cv=False, Kfold=5):
    print(f"Running {model_name} analysis...")
    model.fit(features_train, label_train)
    prediction = model.predict(features_test)
    test_results = prediction - label_test
    hit_rate = sum(test_results == 0)/len(label_test)

    table_row = [model_name, hit_rate, prediction, label_test, prediction - label_test] 

    table_cv_row = []
    if cv: # run cv if flag is up
        cv_predictor = cross_validation_on_model(model, Kfold, all_features[:,nca_idx], all_labels) 
        hit_rate = cv_predictor[0]
        table_cv_row = [f'{model_name} CV', hit_rate, [], label_test, []]        

    return table_row, table_cv_row


######### TK PC #########
# recordingFolder = "C:\BCI_RECORDINGS\\16-08-22\TK\Sub318324886001"
# recordingFolder_2 = "C:\BCI_RECORDINGS\\16-08-22\TK\Sub318324886002"
# recordingFolder = "C:\BCI_RECORDINGS\\16-08-22\RL\Sub316353903002"

######### RL PC #########
recordingFolder = r'C:\\Users\\Latzres\Desktop\\project\\Recordings\\31-08-22\\TK\Sub318324886002'
# recordingFolder_2 = r'C:\\Users\\Latzres\Desktop\\project\\Recordings\\29-08-22\\TK\Sub318324886002'
# recordingFolder = r'C:\\Users\\Latzres\Desktop\\project\\Recordings\\31-08-22\\RL\Sub316353903004'
# recordingFolder_2 = r'C:\\Users\\Latzres\Desktop\\project\\Recordings\\30-08-22\\RL\Sub316353903003'
# recordingFolder = r'C:\\Users\\Latzres\Desktop\\project\\Recordings\\16-08-22\\TT\Sub20220816003'

# All of the features before train-test partition
all_features = sio.loadmat(recordingFolder + '\AllDataInFeatures.mat')['AllDataInFeatures']
all_labels = sio.loadmat(recordingFolder + '\\trainingVec.mat')['trainingVec'].ravel()
test_indices = sio.loadmat(recordingFolder + '\\testIdx.mat')['testIdx'].ravel()
nca_selected_idx = sio.loadmat(recordingFolder + '\\SelectedIdx.mat')['SelectedIdx'].ravel() - 1 
print(nca_selected_idx)
print(headers[nca_selected_idx])
if sys.argv[1] == '2':
    all_features_2 = sio.loadmat(recordingFolder_2 + '\AllDataInFeatures.mat')['AllDataInFeatures']
    all_labels_2 = sio.loadmat(recordingFolder_2 + '\\trainingVec.mat')['trainingVec'].ravel()
    test_indices_2 = sio.loadmat(recordingFolder_2 + '\\testIdx.mat')['testIdx'].ravel()
    nca_selected_idx_2 = sio.loadmat(recordingFolder_2 + '\\SelectedIdx.mat')['SelectedIdx'].ravel() - 1
    
    all_features = np.concatenate((all_features, all_features_2), axis=0)
    all_labels = np.concatenate((all_labels, all_labels_2), axis=0)
    test_indices = np.concatenate((test_indices, test_indices_2 + len(test_indices)), axis=0)
    
    nca_selected_idx = np.concatenate((nca_selected_idx[:5], nca_selected_idx_2[:5]), axis=0)
    print(f"concatenated headers: {headers[nca_selected_idx]}")

nca = NCA(n_components=10)
nca_all_features = nca.fit_transform(all_features, all_labels)

print("shapes: ")
print(all_features.shape, all_labels.shape, test_indices.shape, nca_selected_idx.shape, all_features[:,nca_selected_idx].shape)
test_indices = test_indices - 1
train_indices = [i for i in range(len(all_labels)) if i not in test_indices]

#NCA analysis
train_features_nca = nca_all_features[train_indices]
test_features_nca = nca_all_features[test_indices]

labels_train_nca = all_labels[train_indices]
labels_test_nca = all_labels[test_indices]


# GENETIC ALGORITHM analysis
features_train_ga = all_features[train_indices]
features_test_ga = all_features[test_indices]

labels_train_ga = all_labels[train_indices]
labels_test_ga = all_labels[test_indices]

estimator = SVM(penalty='l2', loss='hinge', multi_class='ovr', C=0.1, max_iter=30_000)
selector = GeneticSelectionCV(
    estimator,
    cv = 3,
    scoring = "accuracy",
    max_features = 10,
    n_population = 153,
    crossover_proba = 0.5,
    mutation_proba = 0.2,
    n_generations = 60,
    caching = True,
    mutation_independent_proba = 0.025,
    crossover_independent_proba = 0.8
)

selector = selector.fit(features_train_ga, labels_train_ga)
print(f"SVM GA FEATURES: {headers[selector.support_]}")
svm_ga_prediction = selector.predict(features_test_ga)
test_results = svm_ga_prediction - labels_test_ga
hit_rate = sum(test_results == 0)/len(labels_test_ga)

svm_ga_row = ['SVM GA', hit_rate, svm_ga_prediction, labels_test_ga, svm_ga_prediction - labels_test_ga]

estimator = LDA()
selector = GeneticSelectionCV(
    estimator,
    cv = 3,
    scoring = "accuracy",
    max_features = 10,
    n_population = 153,
    crossover_proba = 0.5,
    mutation_proba = 0.2,
    n_generations = 60,
    caching = True,
    mutation_independent_proba = 0.025,
    crossover_independent_proba = 0.8
)

selector = selector.fit(features_train_ga, labels_train_ga)
print(f"LDA GA FEATURES: {headers[selector.support_]}")
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


##### Running Models Classifications #####
models = [
    {'name': 'LDA', 'model': LDA(), 'cv': True},
    {'name': 'LDA NCA', 'model': LDA(), 'cv': True},
    {'name': 'QDA', 'model': QDA(), 'cv': True},
    {'name': 'QDA NCA', 'model': QDA(), 'cv': True},
    {'name': 'KNN-5', 'model': KNN(5), 'cv': False},
    {'name': 'KNN-5 NCA', 'model': KNN(5), 'cv': False},
    {'name': 'KNN-7', 'model': KNN(7), 'cv': False},
    {'name': 'KNN-7 NCA', 'model': KNN(7), 'cv': False},
    {'name': 'SVM', 'model': SVM(penalty='l2', loss='hinge', multi_class='ovr', C=2, max_iter=30_000), 'cv': True},
    {'name': 'SVM NCA', 'model': SVM(penalty='l2', loss='hinge', multi_class='ovr', C=2, max_iter=30_000), 'cv': True},
    {'name': 'NB', 'model': NB(), 'cv': False},
    {'name': 'NB NCA', 'model': NB(), 'cv': False},
    {'name': 'RF', 'model': RF(criterion='entropy'), 'cv': True},
    {'name': 'RF NCA', 'model': RF(criterion='entropy'), 'cv': True},
    {'name': 'DT', 'model': DT(), 'cv': True},
    {'name': 'DT NCA', 'model': DT(), 'cv': True},
]

all_rows = []

for model in models:
    is_nca = 'NCA' in model['name']
    is_ga = 'GA' in model['name']  # preparing ground
    f_train = features_train if not is_nca else train_features_nca
    f_test = features_test if not is_nca else test_features_nca
    l_train = label_train if not is_nca else labels_train_nca
    l_test = label_test if not is_nca else labels_test_nca
    row, cv_row = classify_results(model['model'], model['name'], features_train=f_train, features_test=f_test, label_train=l_train, nca_idx=nca_selected_idx, label_test=l_test, cv=model['cv'])
    all_rows.append(row)
    if cv_row != []:
        all_rows.append(cv_row)

all_rows.append(svm_ga_row)
all_rows.append(lda_ga_row)

#### ---------- Priniting table ---------- ####
print('')
table_headers = ["Classifier", "Success Rate", "Classifier Prediction", "Test Labels", "Sub Labels"]
print(tabulate(all_rows, headers=table_headers))