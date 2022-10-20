from cProfile import label
import scipy.io as sio
import numpy as np
import argparse
from metrics_wrapper import get_paths
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
from OurFeatureSelection import Selector

# TODO: Add interface for choosing features as we want.
# TODO: How to do the feature mentioned above.
# TODO: Add genetic algorithm feature selection

features_names_list = ['BP_15.5_18.5', 'BP_8_10.5', 'BP_10_15.5', 'BP_17.5_20.5', 'BP_12.5_30', 'RTP', 'SPEC_MOM', 'SPEC_EDGE', 'SPEC_ENT', 'SLOPE', 'INTERCEPT', 'MEAN_FREQ', 'OCC_BAND', 'POWER_BAND', 'WLT_ENT', 'KURT', 'SKEW', 'VAR', 'STD', 'LOG_ENE_ENT', 'BETA_ALPHA_RATIO', 'BP_THETA']
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


def classify_results(model, model_name, features_train, label_train, features_test, label_test, features_indices, cv=False, Kfold=5):
    print(f"Running {model_name} analysis...")
    model.fit(features_train, label_train)
    prediction = model.predict(features_test)
    test_results = prediction - label_test
    hit_rate = sum(test_results == 0)/len(label_test)

    table_row = [model_name, hit_rate, prediction, label_test, prediction - label_test] 

    table_cv_row = []
    if cv: # run cv if flag is up
        cv_predictor = cross_validation_on_model(model, Kfold, all_features[:,features_indices], all_labels) 
        hit_rate = cv_predictor[0]
        table_cv_row = [f'{model_name} CV', hit_rate, [], label_test, []]        

    return table_row, table_cv_row


def classify_results_ga(selection_params, features_train, label_train, features_test, label_test, recordingFolder, cv=False, Kfold=5):
    print(f"Running {selection_params['name']} with GA features selection & analysis...")
    selector = GeneticSelectionCV(
        selection_params['model'],
        cv = selection_params['cv'],
        scoring = selection_params['scoring'],
        max_features = selection_params['max_features'],
        n_population = selection_params['n_population'],
        crossover_proba = selection_params['cross_prob'],
        mutation_proba = selection_params['muta_prob'],
        n_generations = selection_params['n_gens'],
        caching = selection_params['caching'],
        mutation_independent_proba = selection_params['muta_ind_prob'],
        crossover_independent_proba = selection_params['cross_ind_prob']
    )
    selector = selector.fit(features_train, label_train)
    np.savetxt(f'{recordingFolder}\{selection_params["name"]}_ga_features.txt', headers[selector.support_], fmt='%s')

    prediction = selector.predict(features_test)
    test_results = prediction - label_test
    hit_rate = sum(test_results == 0)/len(label_test)

    row = [f'{selection_params["name"]} GA', hit_rate, prediction, label_test, prediction - label_test]

    cv_row = []
    if cv:
        cv_prediction = cross_validation_on_model(selection_params['model'], Kfold, all_features[:,selector.support_], all_labels)
        hit_rate = cv_prediction[0]
        cv_row = [f'{selection_params["name"]} GA CV', hit_rate, [], label_test, []]

    return row, cv_row


def get_dict_for_folder_from_path(path):
    list_of_path = path.split('\\')
    return {"name": list_of_path[-2], "date": list_of_path[-3], "num": int(list_of_path[-1][-2:])}


def parse_cmdl():
    parser = argparse.ArgumentParser(description='This script is running classifiers on the requested folder, and then produces a table and csv file to compare between \n various classifiers and feature selection methods.')
    parser.add_argument('--folder', '-f', dest='folder', help='Folder path of the recording we want to classify', type=str)
    parser.add_argument('--folder2', '-f2', dest='folder2', help='Folder path of the second recording we want to classify - taken into account only if unify is True', type=str, default=None)
    parser.add_argument('--unify', '-u', dest='unify', help='Unify folder and folder2 to "one" recording and classify', type=bool)
    parser.add_argument('--paths', '-p', dest='paths', help='Path to paths file to run classisfication for each folder in paths file', type=str, default=None)
    parser.add_argument('--metric', '-m', dest='metric', help='Metric string - according to him we\'ll choose our features by our statistic methods', type=str, default="Score_(R^2)_Left,Score_(R^2)_Right")
    parser.add_argument('--simple', '-s', dest='simple', help='Use simple metric (just sort dataframe by columns)', type=bool, default=True)
    args = parser.parse_args()
    return {'folder': args.folder,
            'folder2': args.folder2,
            'unify': args.unify,
            'paths': args.paths,
            'metric': args.metric,
            'simple': args.simple}


def get_matlab_features(recordingFolder, recordingFolder_2, unify):
    # features from matlab neighborhood component analysis - takes 10 best features.
    features_train = sio.loadmat(recordingFolder + '\FeaturesTrainSelected.mat')['FeaturesTrainSelected']
    label_train = sio.loadmat(recordingFolder + '\LabelTrain.mat')['LabelTrain'].ravel()

    features_test = sio.loadmat(recordingFolder + '\FeaturesTest.mat')['FeaturesTest']
    label_test = sio.loadmat(recordingFolder + '\LabelTest.mat')['LabelTest'].ravel()

    if recordingFolder_2 is not None and unify:
        features_train_2 = sio.loadmat(recordingFolder_2 + '\FeaturesTrainSelected.mat')['FeaturesTrainSelected']
        label_train_2 = sio.loadmat(recordingFolder_2 + '\LabelTrain.mat')['LabelTrain'].ravel()

        features_test_2 = sio.loadmat(recordingFolder_2 + '\FeaturesTest.mat')['FeaturesTest']
        label_test_2 = sio.loadmat(recordingFolder_2 + '\LabelTest.mat')['LabelTest'].ravel()
        
        features_train = np.concatenate((features_train, features_train_2), axis=0)
        label_train = np.concatenate((label_train, label_train_2), axis=0)
        features_test = np.concatenate((features_test, features_test_2), axis=0)
        label_test = np.concatenate((label_test, label_test_2), axis=0)

    return features_train, label_train, features_test, label_test


def get_all_features(recordingFolder, recordingFolder_2, unify):
    all_features = sio.loadmat(recordingFolder + '\AllDataInFeatures.mat')['AllDataInFeatures']
    all_labels = sio.loadmat(recordingFolder + '\\trainingVec.mat')['trainingVec'].ravel()
    test_indices = sio.loadmat(recordingFolder + '\\testIdx.mat')['testIdx'].ravel()
    nca_selected_idx = sio.loadmat(recordingFolder + '\\SelectedIdx.mat')['SelectedIdx'].ravel() - 1 
    print(nca_selected_idx)
    print(headers[nca_selected_idx])
    if recordingFolder_2 is not None and unify:
        all_features_2 = sio.loadmat(recordingFolder_2 + '\AllDataInFeatures.mat')['AllDataInFeatures']
        all_labels_2 = sio.loadmat(recordingFolder_2 + '\\trainingVec.mat')['trainingVec'].ravel()
        test_indices_2 = sio.loadmat(recordingFolder_2 + '\\testIdx.mat')['testIdx'].ravel()
        nca_selected_idx_2 = sio.loadmat(recordingFolder_2 + '\\SelectedIdx.mat')['SelectedIdx'].ravel() - 1
        
        all_features = np.concatenate((all_features, all_features_2), axis=0)
        all_labels = np.concatenate((all_labels, all_labels_2), axis=0)
        test_indices = np.concatenate((test_indices, test_indices_2 + len(test_indices)), axis=0)
        
        nca_selected_idx = np.concatenate((nca_selected_idx[:5], nca_selected_idx_2[:5]), axis=0)
        print(f"concatenated headers: {headers[nca_selected_idx]}")
    return all_features, all_labels, test_indices, nca_selected_idx


def classify(args_dict):

    recordingFolder = args_dict['folder']
    recordingFolder_2 = args_dict['folder2']

    # All of the features before train-test partition
    global all_features
    global all_labels
    all_features, all_labels, test_indices, nca_selected_idx = get_all_features(recordingFolder, recordingFolder_2, args_dict['unify'])
    
    nca = NCA(n_components=10)
    nca_all_features = nca.fit_transform(all_features, all_labels)
    print("shapes: ")
    print(all_features.shape, all_labels.shape, test_indices.shape, nca_selected_idx.shape, all_features[:,nca_selected_idx].shape)
    test_indices = test_indices - 1
    train_indices = [i for i in range(len(all_labels)) if i not in test_indices]

    #### ------------ NCA analysis ------------ ####
    train_features_nca = nca_all_features[train_indices]
    test_features_nca = nca_all_features[test_indices]

    labels_train_nca = all_labels[train_indices]
    labels_test_nca = all_labels[test_indices]


    #### ------------ GENETIC ALGORITHM analysis ------------ ####
    features_train_ga = all_features[train_indices]
    features_test_ga = all_features[test_indices]

    labels_train_ga = all_labels[train_indices]
    labels_test_ga = all_labels[test_indices]

    ga_models = [
        {'name': 'SVM', 
         'model': SVM(penalty='l2', loss='hinge', multi_class='ovr', C=0.1, max_iter=30_000),
         'cv': 3,
         "scoring": "accuracy",
         "max_features": 10,
         "n_population": 153,
         "cross_prob": 0.5,
         "muta_prob": 0.2,
         "n_gens": 60,
         "caching": True,
         "muta_ind_prob": 0.025,
         "cross_ind_prob": 0.8 }, 
        
        {'name': 'LDA', 
         'model': LDA(),
         'cv': 3,
         "scoring": "accuracy",
         "max_features": 10,
         "n_population": 153,
         "cross_prob": 0.5,
         "muta_prob": 0.2,
         "n_gens": 60,
         "caching": True,
         "muta_ind_prob": 0.025,
         "cross_ind_prob": 0.8 },

    ]

    #### ------------ features from matlab neighborhood component analysis - takes 10 best features ------------ #
    features_train, label_train, features_test, label_test = get_matlab_features(recordingFolder, recordingFolder_2, args_dict['unify']) 

    #### ------------ features from statistical analysis ------------ ####
    our_selector = Selector('paths/paths_TK.txt', record_path=recordingFolder, ascending=False)
    if args_dict['simple']:
        our_features_indices = our_selector.select_features(args_dict['metric'].split(','), use_prior=False)
    else:
        our_features_indices = our_selector.select_features(args_dict['metric'], use_prior=False, simple_rule=False)
    
    train_features_stats = all_features[train_indices][:,our_features_indices]
    test_features_stats = all_features[test_indices][:,our_features_indices]

    labels_train_stats = all_labels[train_indices]
    labels_test_stats = all_labels[test_indices]

    ##### ------------ Running Models Classifications ------------ #####
    models = [
        {'name': 'LDA', 'model': LDA(), 'cv': True},
        {'name': 'LDA NCA', 'model': LDA(), 'cv': True, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca},
        {'name': 'LDA STA', 'model': LDA(), 'cv': True, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats},
        {'name': 'QDA', 'model': QDA(), 'cv': True},
        {'name': 'QDA NCA', 'model': QDA(), 'cv': True, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca},
        {'name': 'QDA STA', 'model': QDA(), 'cv': True, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats},
        {'name': 'KNN-5', 'model': KNN(5), 'cv': False},
        {'name': 'KNN-5 NCA', 'model': KNN(5), 'cv': False, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca},
        {'name': 'KNN-5 STA', 'model': KNN(5), 'cv': False, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats},
        {'name': 'KNN-7', 'model': KNN(7), 'cv': False},
        {'name': 'KNN-7 NCA', 'model': KNN(7), 'cv': False, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca},
        {'name': 'KNN-7 STA', 'model': KNN(7), 'cv': False, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats},
        {'name': 'SVM', 'model': SVM(penalty='l2', loss='hinge', multi_class='ovr', C=2, max_iter=30_000), 'cv': True},
        {'name': 'SVM NCA', 'model': SVM(penalty='l2', loss='hinge', multi_class='ovr', C=2, max_iter=30_000), 'cv': True, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca},
        {'name': 'SVM STA', 'model': SVM(penalty='l2', loss='hinge', multi_class='ovr', C=2, max_iter=30_000), 'cv': True, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats},
        {'name': 'NB', 'model': NB(), 'cv': False},
        {'name': 'NB NCA', 'model': NB(), 'cv': False, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca},
        {'name': 'NB STA', 'model': NB(), 'cv': False, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats},
        {'name': 'RF', 'model': RF(criterion='entropy'), 'cv': True},
        {'name': 'RF NCA', 'model': RF(criterion='entropy'), 'cv': True, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca},
        {'name': 'RF STA', 'model': RF(criterion='entropy'), 'cv': True, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats},
        {'name': 'DT', 'model': DT(), 'cv': True},
        {'name': 'DT NCA', 'model': DT(), 'cv': True, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca},
        {'name': 'DT STA', 'model': DT(), 'cv': True, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats},
    ]

    all_rows = []

    for model in models:
        f_train = features_train if model.get('ftr') is None else model.get('ftr')
        f_test = features_test if model.get('fte') is None else model.get('fte')
        l_train = label_train if model.get('ltr') is None else model.get('ltr')
        l_test = label_test if model.get('lte') is None else model.get('lte')
        indices = nca_selected_idx if model.get('indices') is None else model.get('indices')
        row, cv_row = classify_results(model['model'], model['name'], features_train=f_train, features_test=f_test, label_train=l_train, features_indices=indices, label_test=l_test, cv=model['cv'])
        all_rows.append(row)
        if cv_row != []:
            all_rows.append(cv_row)

    for model in ga_models:
        row, cv_row = classify_results_ga(model, features_train_ga, labels_train_ga, features_test_ga, labels_test_ga, recordingFolder, cv=True)
        all_rows.append(row)
        all_rows.append(cv_row)

    #### ---------- Priniting table ---------- ####
    print('')
    table_headers = ["Classifier", "Success Rate", "Classifier Prediction", "Test Labels", "Sub Labels"]
    print(tabulate(all_rows, headers=table_headers))

    folder_dict = get_dict_for_folder_from_path(recordingFolder)
    all_rows.insert(0, table_headers)
    np.savetxt(f'class_results/{folder_dict["name"]}_{folder_dict["date"]}_{folder_dict["num"]}.csv', np.array(all_rows, dtype=object), delimiter=',', fmt='%s')

if __name__ == '__main__':
    args_dict = parse_cmdl()
    if args_dict['paths'] is None:
        classify(args_dict)
    else:
        paths = get_paths(paths_file=args_dict['paths'])
        for path in paths:
            args_dict['folder'] = path
            print(f'running now classify on path: {path}')
            classify(args_dict)
            
