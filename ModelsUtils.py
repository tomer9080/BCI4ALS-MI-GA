import numpy as np
from genetic_selection import GeneticSelectionCV
from sklearn.model_selection import KFold
from sklearn_genetic import GAFeatureSelectionCV
from sklearn.metrics import accuracy_score
import OurUtils as Utils
import copy

features_names_list = Utils.features_names_list
headers = Utils.headers
from_name_to_index = Utils.from_feature_name_to_index

def reduce_ga_search_space(features: np.ndarray, model_name):
    hists_ga: dict = Utils.load_from_pickle('stats\\ga_models_features_hists')
    hist: dict = hists_ga[model_name]
    mask = [feature in hist.keys() for feature in headers]
    return features[:,mask]


# CROSS-VALIDATION
def cross_validation_on_model(model, k, features, labels, mv=False, nca_indicies=None):
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

    all_scores = []
    all_models = []
    for train_index, test_index in kf.split(features):
        X_train = features[train_index]
        X_test = features[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
            
        #Train the model
        if not mv:
            model.fit(X_train, y_train) #Training the model
            score = accuracy_score(y_test, model.predict(X_test))
            all_scores.append(score)
            all_models.append(model)
        else:
            taken_models = {}
            for key, classifier in model.items():
                if 'GA' in key:
                    classifier.fit(X_train, y_train) #Training the model
                else:
                    classifier.fit(X_train[:,nca_indicies], y_train) #Training the model
                taken_models[key] = classifier
            score_list = classify_majority(None, taken_models, features, labels, test_index, nca_indicies)
            all_scores.append(score_list[1])
            all_models.append(taken_models)
        
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

    return table_row, table_cv_row


def classify_results_ga(selection_params, features_train, label_train, features_test, label_test, recordingFolder, folder_dict, cv=False, Kfold=5, unify=False, chosen_indices={}, all_features=[], all_labels=[], mv_dict: dict={}):
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
    reduced_features = reduce_ga_search_space(features=features_train, model_name=selection_params['name'])
    print(reduced_features.shape)
    selector = selector.fit(reduced_features, label_train)
    chosen_indices[selection_params["name"]] = np.array([i for i, res in enumerate(selector.support_) if res == True])
    # np.savetxt(f'{recordingFolder}\{selection_params["name"]}_ga_features.txt', headers[selector.support_], fmt='%s')
    # Utils.create_sub_folder_for_ga_features(f'{selection_params["name"]}')
    # np.savetxt(f'ga_features\\{selection_params["name"]}\\{folder_dict["name"]}_{folder_dict["date"]}_{folder_dict["num"]}_ga_features.txt',  headers[selector.support_], fmt='%s')
    
    reduced_features_test = reduce_ga_search_space(features_test, model_name=selection_params['name'])
    prediction = selector.predict(reduced_features_test)
    test_results = prediction - label_test
    hit_rate = sum(test_results == 0)/len(label_test)

    mv_dict['MV_GA'][selection_params['name'] + ' GA'] = copy.deepcopy(selector)

    if unify:
        row = [f'{selection_params["name"]} GA', hit_rate, prediction, label_test]
    else:
        row = [f'{selection_params["name"]} GA', hit_rate, prediction, label_test, prediction - label_test]

    cv_row = []
    reduced_all_features = reduce_ga_search_space(all_features.copy(), selection_params['name'])
    if cv:
        cv_prediction = cross_validation_on_model(selection_params['model'], Kfold, reduced_all_features[:,selector.support_], all_labels)
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

def classify_online_model(offline_model, model_name, features_indices, all_features, all_labels):
    print(f"Running online classification! {model_name}")
    offline_prediction = offline_model.predict(all_features[:,features_indices])
    result = offline_prediction - all_labels
    hit_rate = sum(result == 0)/len(all_labels)

    return [model_name, hit_rate, offline_prediction, all_labels, offline_prediction - all_labels]


def classify_majority(key_name, models: dict, features, labels, test_indices, nca_indices):
    label_test = labels[test_indices]
    taken_models = {}
    for key, model in models.items():
        features_test = features[:,nca_indices]
        features_test = features_test[test_indices,:]
        print(key)
        if any([model_name in key for model_name in ['STA', 'QDA', 'DT']]):
            continue
        if 'GA' in key:
            features_test = features[test_indices,:]
            print("To be GA or not to be")
        score = accuracy_score(label_test, model.predict(features_test))
        if score > 0.4:
            taken_models[key] = model.predict_proba(features_test)
    
    # matrices_sum = np.sum(np.array(list(taken_models.values())), axis=0)
    matrices_mul = np.ones((15,3))
    for key, matrix in taken_models.items():
        matrices_mul *= matrix
    matrices_sum = matrices_mul
    decision_matrix: np.ndarray = matrices_sum / len(taken_models.keys())
    prediction = list(np.argmax(decision_matrix, axis=1) + 1)
    hit_rate = sum(prediction - label_test == 0) / len(label_test)
    row = [key_name, hit_rate, prediction, label_test, prediction - label_test]
    
    return row

def classify_ensemble(key_name, models: dict, features, labels, test_indices, nca_indices, eta=0.2):
    weights = {key: 1 for key in models.keys()} # initializing all weights to 1
    prob_matrices = {}
    train_indices = [i for i in range(len(labels)) if i not in test_indices]
    # Zip all matrices - work line by line to be each prediction.
    # This is "training" the model
    for key, model in models.items():
        if 'GA' in key:
            print(f'Here! :{features.shape}')
            train_features = reduce_ga_search_space(features, key.replace(' GA', ''))
            train_features = train_features[train_indices,:]
        else:
            train_features = features[:,nca_indices]
            train_features = train_features[train_indices,:]
        prob_matrices[key] = model.predict_proba(train_features)
    for j, i in enumerate(train_indices):
        y_true = labels[i]
        for key, matrix in prob_matrices.items():
            prediction = np.argmax(matrix[j,:]) + 1
            # print(f"{key}: {(prediction, y_true)}")
            if prediction != y_true:
                weights[key] *= (1 - eta)
    print(weights)
    # return a prediction
    final_proba_matrix = np.zeros((len(test_indices), 3))
    for key, model in models.items():
        if 'GA' in key:
            features_test = reduce_ga_search_space(features, key.replace(' GA', ''))
            features_test = features_test[test_indices,:]
        else:
            features_test = features[:,nca_indices]
            features_test = features_test[test_indices,:]
        final_proba_matrix += (weights[key] * model.predict_proba(features_test))
    prediction = list(np.argmax(final_proba_matrix, axis=1) + 1)
    label_test = labels[test_indices]
    hit_rate = sum(prediction - label_test == 0) / len(label_test)
    row = [key_name, hit_rate, prediction, label_test, prediction - label_test]

    return row