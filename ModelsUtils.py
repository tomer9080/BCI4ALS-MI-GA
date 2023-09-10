import numpy as np
from genetic_selection import GeneticSelectionCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import OurUtils as Utils
import copy
import os
from Stacking import Stacking
from GAModel import GAModel
from EA import EA
from Parsers import get_args_dict

features_names_list = Utils.features_names_list
headers = Utils.headers
from_name_to_index = Utils.from_feature_name_to_index

def reduce_ga_search_space(features: np.ndarray, model_name, thresh=None):
    threshold = thresh if thresh else get_args_dict().get('threshold', 0)
    print("Wallak threshold is {}".format(threshold))
    hists_ga: dict = Utils.load_from_pickle(os.path.join('stats', 'ga_models_features_hists'))
    hist: dict = hists_ga.get(model_name)
    if hist is None:
        return features, [True] * len(headers)
    mask = [(feature in hist.keys() and hist.get(feature, 0) >= threshold) for feature in headers]
    return features[:,mask], mask


# CROSS-VALIDATION
def cross_validation_on_model(model, k, features, labels, mv=False, nca_indicies=None, postfix=''):
    """
    cross_validation_on_model - given a model, runs a k-fold CV on him, and return a 
    tuple (avg_score, all_scores, all_models)
    :param - model: the model we want to CV
    :param - k: k fold parameter
    :param - features: the features to train the model on
    :param - labels: the label to train the model on
    :return: tuple (avg_score, all_scores, all_models)
    avg_score - the mean score from all fold predictions
    all_scores - all of the scores for each fold
    all_models - all the models that has been trained on the cv session.
    """
    kf = KFold(n_splits=k, shuffle=False)

    all_scores = []
    all_models = []
    for train_index, test_index in kf.split(features):
        X_train = features[train_index]
        X_test = features[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
            
        #Train the model
        if not mv: # single model
            model.fit(X_train, y_train) #Training the model
            score = accuracy_score(y_test, model.predict(X_test))
            all_scores.append(score)
            all_models.append(model)
        else: # ensembles - let's return a 6 rows matrix
            scores = {'MV CV': [], 'MV GA CV': [], 'MV ALL CV': [], 'OSTACKING CV': [], 'OSTACKING GA CV': [], 'OSTACKING ALL CV': []}
            taken_models = {}
            for key, classifier in model.items():
                if 'GA' in key:
                    classifier.fit(X_train, y_train)  # Training the model
                else:
                    classifier.fit(X_train[:,nca_indicies], y_train)  # Training the model
                taken_models[key] = classifier
            
            # Use trained models to evaluate
            taken_models_reg = {key: val for key, val in taken_models.items() if 'GA' not in key}
            taken_models_ga = {key: val for key, val in taken_models.items() if 'GA' in key}
            scores['MV CV'].append(classify_ensemble(f'MV CV', taken_models_reg, features, labels, test_index, nca_indicies)[1])
            scores['MV GA CV'].append(classify_ensemble(f'MV GA CV', taken_models_ga, features, labels, test_index, nca_indicies)[1])
            scores['MV ALL CV'].append(classify_ensemble(f'MV ALL CV', taken_models, features, labels, test_index, nca_indicies)[1])
            scores['OSTACKING CV'].append(our_classify_stacking(f'OSTACKING CV', taken_models_reg, features, labels, test_index, nca_indicies)[1])
            scores['OSTACKING GA CV'].append(our_classify_stacking(f'OSTACKING GA CV', taken_models_ga, features, labels, test_index, nca_indicies)[1])
            scores['OSTACKING ALL CV'].append(our_classify_stacking(f'OSTACKING ALL CV', taken_models, features, labels, test_index, nca_indicies)[1])
            
            scores = {key: np.mean(val) for key, val in scores.items()}
            return scores
            
    avg_score = np.average(all_scores)
    print(f"All scores: {all_scores}")
    return avg_score, all_scores, all_models


def classify_results(model, model_name, features_train, label_train, features_test, label_test, features_indices, cv=False, Kfold=5, params=None, all_features=[], all_labels=[], args: dict={}, mv_dict: dict={}):
    print(f"Running {model_name} analysis...")
    model.fit(features_train, label_train)
    
    prediction = model.predict(features_test)
    test_results = prediction - label_test
    hit_rate = sum(test_results == 0)/len(label_test)

    Utils.save_to_pickle(model, f'{model_name}_object.pkl', args=args)

    mv_dict['MV'][model_name] = copy.deepcopy(model)

    if args.get('unify', False):
        table_row = [model_name, hit_rate, prediction, label_test]
    else: 
        table_row = [model_name, hit_rate, prediction, label_test, prediction - label_test] 

    table_cv_row = []
    if cv: # run cv if flag is up
        cv_predictor = cross_validation_on_model(model, Kfold, all_features[:,features_indices], all_labels) 
        hit_rate = cv_predictor[0]
        table_cv_row = [f'{model_name} CV', hit_rate, [], label_test, []]        

    return table_row[:2], table_cv_row[:2]


def classify_results_ga(selection_params, features_train, label_train, features_test, label_test, recordingFolder, folder_dict, cv=False, Kfold=5, unify=False, chosen_indices={}, all_features=[], all_labels=[], mv_dict: dict={}):
    print(f"Running {selection_params['name']} with GA features selection & analysis...")
    selector = GAModel(selection_params['name'], selection_params['model'])
    
    selector = selector.fit(features_train, label_train)
    chosen_indices[selection_params["name"]] = np.array([i for i, res in enumerate(selector.support_) if res == True])
    
    Utils.create_sub_folder_for_ga_features(selection_params["name"], selection_params['index_max'])
    np.savetxt(os.path.join('ga_features', f'ga_run_{selection_params["index_max"]}', selection_params["name"], f'{folder_dict["name"]}_{folder_dict["date"]}_ {folder_dict["num"]}_ga_features.txt'), headers[selector.mask][selector.support_], fmt='%s')
    
    prediction = selector.predict(features_test)
    test_results = prediction - label_test
    hit_rate = sum(test_results == 0)/len(label_test)

    mv_dict['MV_GA'][selection_params['name'] + ' GA'] = copy.deepcopy(selector)

    if unify:
        row = [f'{selection_params["name"]} GA', hit_rate, prediction, label_test]
    else:
        row = [f'{selection_params["name"]} GA', hit_rate, prediction, label_test, prediction - label_test]

    cv_row = []
    reduced_all_features, _ = reduce_ga_search_space(all_features.copy(), selection_params['name'])
    if cv:
        cv_prediction = cross_validation_on_model(selection_params['model'], Kfold, reduced_all_features[:,selector.support_], all_labels)
        hit_rate = cv_prediction[0]
        cv_row = [f'{selection_params["name"]} GA CV', hit_rate, [], label_test, []]

    return row[:2], cv_row[:2]


def classify_online_model(offline_model, model_name, features_indices, all_features, all_labels):
    print(f"Running online classification! {model_name}")
    offline_prediction = offline_model.predict(all_features[:,features_indices])
    result = offline_prediction - all_labels
    hit_rate = sum(result == 0)/len(all_labels)

    return [model_name, hit_rate, offline_prediction, all_labels, offline_prediction - all_labels][:2]

def get_train_indices(test_indices, labels):
    return [i for i in range(len(labels)) if i not in test_indices]

def classify_ensemble(key_name, models: dict, features, labels, test_indices, nca_indices, eta=0.2):
    ea_model = EA(models, test_indices, nca_indices, labels, features, eta)
    ea_model.fit()
    return ea_model.predict(key_name)

def mv_cv(key_name, models: dict, features, labels, test_indices, nca_indices, eta=0.2, folds=5):
    ea_model = EA(models, test_indices, nca_indices, labels, features, eta)
    cv_score = ea_model.run_cv(folds=folds)
    return [key_name, cv_score]

def make_stacking_model(models: dict):
    level0 = list(models.items())
    level1 = LogisticRegression()
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model

def classify_stacking(key_name, models: dict, features, labels, test_indices, nca_indices):
    stacking_model = make_stacking_model(models)
    train_indices = get_train_indices(test_indices, labels)
    stacking_model.fit(features[train_indices,:][:,nca_indices], labels[train_indices])
    prediction = stacking_model.predict((features[test_indices,:])[:,nca_indices])
    label_test = labels[test_indices]
    return row_to_print(prediction, label_test, key_name)

def our_classify_stacking(key_name, models: dict, features, labels, test_indices, nca_indices):
    stacking_model = Stacking(models, test_indices, nca_indices, labels, features, LogisticRegression())
    stacking_model.fit()
    prediction = stacking_model.predict()
    label_test = labels[test_indices]
    return row_to_print(prediction, label_test, key_name)

def stacking_cv(key_name, models: dict, features, labels, test_indices, nca_indices, folds=5):
    stacking_model = Stacking(models, test_indices, nca_indices, labels, features, LogisticRegression())
    cv_score = stacking_model.run_cv(folds=folds)
    return [key_name, cv_score]

def row_to_print(prediction, label_test, key_name):
    hit_rate = sum(prediction - label_test == 0) / len(label_test)
    row = [key_name, hit_rate, prediction, label_test, prediction - label_test]

    return row[:2]
