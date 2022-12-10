import numpy as np
from genetic_selection import GeneticSelectionCV
from sklearn.model_selection import KFold
from sklearn_genetic import GAFeatureSelectionCV
from sklearn.metrics import accuracy_score
import OurUtils as Utils

features_names_list = Utils.features_names_list
headers = Utils.headers

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


def classify_results(model, model_name, features_train, label_train, features_test, label_test, features_indices, cv=False, Kfold=5, params=None, unify=False, all_features=[], all_labels=[]):
    print(f"Running {model_name} analysis...")
    fitted = model.fit(features_train, label_train)
    prediction = model.predict(features_test)
    test_results = prediction - label_test
    hit_rate = sum(test_results == 0)/len(label_test)

    Utils.save_to_pickle(model, f'tmp/{model_name}_object.pkl')

    if unify:
        table_row = [model_name, hit_rate, prediction, label_test]
    else: 
        table_row = [model_name, hit_rate, prediction, label_test, prediction - label_test] 

    table_cv_row = []
    if cv: # run cv if flag is up
        cv_predictor = cross_validation_on_model(model, Kfold, all_features[:,features_indices], all_labels) 
        hit_rate = cv_predictor[0]
        table_cv_row = [f'{model_name} CV', hit_rate, [], label_test, []]        

    return table_row, table_cv_row


def classify_results_ga(selection_params, features_train, label_train, features_test, label_test, recordingFolder, folder_dict, cv=False, Kfold=5, unify=False, chosen_indices={}, all_features=[], all_labels=[]):
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
    chosen_indices[selection_params["name"]] = np.array([i for i, res in enumerate(selector.support_) if res == True])
    np.savetxt(f'{recordingFolder}\{selection_params["name"]}_ga_features.txt', headers[selector.support_], fmt='%s')
    Utils.create_sub_folder_for_ga_features(selection_params["name"])
    np.savetxt(f'ga_features\\{selection_params["name"]}\\{folder_dict["name"]}_{folder_dict["date"]}_{folder_dict["num"]}_ga_features.txt',  headers[selector.support_], fmt='%s')
        

    prediction = selector.predict(features_test)
    test_results = prediction - label_test
    hit_rate = sum(test_results == 0)/len(label_test)

    if unify:
        row = [f'{selection_params["name"]} GA', hit_rate, prediction, label_test]
    else:
        row = [f'{selection_params["name"]} GA', hit_rate, prediction, label_test, prediction - label_test]

    cv_row = []
    if cv:
        cv_prediction = cross_validation_on_model(selection_params['model'], Kfold, all_features[:,selector.support_], all_labels)
        hit_rate = cv_prediction[0]
        cv_row = [f'{selection_params["name"]} GA CV', hit_rate, [], label_test, []]

    return row, cv_row


def classify_results_ga_sklearn(selection_params, features_train, label_train, features_test, label_test, recordingFolder, cv=False, Kfold=5, unify=False, chosen_indices={}, all_features=[], all_labels=[]):
    print(f"Running {selection_params['name']} with GA features selection & analysis...")
    selector = GAFeatureSelectionCV(
        selection_params['model'],
        cv = selection_params['cv'],
        scoring = selection_params['scoring'],
        max_features = selection_params['max_features'],
        population_size = selection_params['n_population'],
        crossover_probability = selection_params['cross_prob'],
        mutation_probability = selection_params['muta_prob'],
        generations = selection_params['n_gens'],
        n_jobs=2
    )
    # del globals()['Individual']
    print(globals())
    selector = selector.fit(features_train, label_train)
    chosen_indices[selection_params["name"]] = np.array([i for i, res in enumerate(selector.best_features_) if res == True])
    np.savetxt(f'{recordingFolder}\{selection_params["name"]}_ga_features_sklearn.txt', headers[selector.best_features_], fmt='%s')

    prediction = selector.predict(features_test[:,selector.best_features_])
    test_results = prediction - label_test
    hit_rate = sum(test_results == 0)/len(label_test)

    if unify:
        row = [f'{selection_params["name"]} GA SK', hit_rate, prediction, label_test]
    else:
        row = [f'{selection_params["name"]} GA SK', hit_rate, prediction, label_test, prediction - label_test]

    cv_row = []
    if cv:
        cv_prediction = cross_validation_on_model(selection_params['model'], Kfold, all_features[:,selector.best_features_], all_labels)
        hit_rate = cv_prediction[0]
        cv_row = [f'{selection_params["name"]} GA CV', hit_rate, [], label_test, []]

    return row, cv_row


def classify_results_gs(model, model_name, features_train, label_train, features_test, label_test, grid, unify=False):
    print(f"Running {model_name} analysis...")
    grid_result = grid.fit(features_train, label_train)
    best_model = grid.best_estimator_
    Utils.save_best_model_stats(model_name, grid_result)

    prediction = best_model.predict(features_test)
    test_results = prediction - label_test
    hit_rate = sum(test_results == 0)/len(label_test)

    if unify:
        table_row = [model_name, hit_rate, prediction, label_test]
    else: 
        table_row = [model_name, hit_rate, prediction, label_test, prediction - label_test] 

    return table_row

