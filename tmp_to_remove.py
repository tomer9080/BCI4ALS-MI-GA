
# Maybe to be demolished since sklearn GA won't work:
"""
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

"""
# old classify majority. not good.
"""
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

"""