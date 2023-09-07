import numpy as np
import Parsers
import pickle
import OurUtils as Utils
import ModelsUtils
import FeaturesExpansion
# from MetricsFeatureSelection import Selector
from tabulate import tabulate
from ModelsParams import build_ga_models, build_models
from GridSearchParams import build_gs_models
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA


features_names_list = Utils.features_names_list
headers = Utils.headers

chosen_indices = {}

def classify(args_dict):

    recordingFolder = args_dict['folder']
    recordingFolder_2 = args_dict['folder2']

    # All of the features before train-test partition
    global all_features
    global all_labels
    all_features, all_labels, test_indices, nca_selected_idx = Utils.get_all_features(recordingFolder, recordingFolder_2, args_dict['unify'])
    test_indices = test_indices - 1
    train_indices = [i for i in range(len(all_labels)) if i not in test_indices]

    num_features = 10
    if args_dict["expanded"]:
        num_features = 15
        all_features = FeaturesExpansion.expand_features(recordingFolder)

    #### ------------ Labels ------------ ####
    labels_train_ga = label_train = all_labels[train_indices]
    labels_test_ga = label_test = all_labels[test_indices]

    #### ------------ GENETIC ALGORITHM analysis ------------ ####
    features_train_ga = all_features[train_indices]
    features_test_ga = all_features[test_indices]

    #### ------------ features from matlab neighborhood component analysis - takes 10 best features ------------ #
    features_train, _, features_test, _ = Utils.get_matlab_features(recordingFolder, recordingFolder_2, args_dict['unify']) 
    chosen_indices["MATLAB"] = nca_selected_idx

    #### ------------ features from statistical analysis ------------ ####
    # file_path = args_dict['paths']
    # our_selector = Selector(file_path, record_path=recordingFolder, ascending=args_dict["ascending"], corr=args_dict["corr"])
    # should_use_prior = False if args_dict['prior'] == 0 else True
    # if args_dict['simple'] == 1:
    #     our_features_indices = our_selector.select_features(args_dict['metric'].split(','), use_prior=should_use_prior, prior_recordings=args_dict['prior'], num_of_features=num_features)
    # else:
    #     our_features_indices = our_selector.select_features(args_dict['metric'], use_prior=should_use_prior, prior_recordings=args_dict['prior'], simple_rule=False, num_of_features=num_features)
    
    # train_features_stats = all_features[train_indices][:,our_features_indices]
    # test_features_stats = all_features[test_indices][:,our_features_indices]

    # labels_train_stats = all_labels[train_indices]
    # labels_test_stats = all_labels[test_indices]

    # chosen_indices["STA"] = np.array(our_features_indices)

    ##### ------------ Running Models Classifications ------------ #####
    folder_dict = Utils.get_dict_for_folder_from_path(recordingFolder)
    # models = build_models(our_features_indices, train_features_stats, test_features_stats, labels_train_stats, labels_test_stats)
    models = build_models()
    ga_models = build_ga_models(15)

    all_rows = []

    major_dict = {'MV': {}, 'MV_GA': {}, 'MV_ALL': {}}
    ##### ============= NCA ANALYSIS ( MATLAB & PYTHON ) ============= #####
    print('started models analysis\n')
    for model in models:
        f_train = features_train if model.get('ftr') is None else model.get('ftr')
        f_test = features_test if model.get('fte') is None else model.get('fte')
        l_train = label_train if model.get('ltr') is None else model.get('ltr')
        l_test = label_test if model.get('lte') is None else model.get('lte')
        indices = nca_selected_idx if model.get('indices') is None else model.get('indices')
        row, cv_row = ModelsUtils.classify_results(model['model'], model['name'], features_train=f_train, features_test=f_test, label_train=l_train, features_indices=indices, label_test=l_test, cv=model['cv'], params=model.get('params'), all_features=all_features, all_labels=all_labels, args=args_dict, mv_dict=major_dict)
        all_rows.append(row)
        if cv_row != []:
            all_rows.append(cv_row)

    ##### ============= GA ANALYSIS ============= #####
    if args_dict['ga']:
        print('started GA models analysis\n')
        for model in ga_models:
            model['index_max'] = args_dict['index_max']
            row, cv_row = ModelsUtils.classify_results_ga(model, features_train_ga, labels_train_ga, features_test_ga, labels_test_ga, recordingFolder, folder_dict, cv=True, chosen_indices=chosen_indices, all_features=all_features, all_labels=all_labels, mv_dict=major_dict)
            all_rows.append(row)
            all_rows.append(cv_row)

    ##### ============= RUN Majority Vote ============= #####
    major_dict['MV_ALL'] = {**major_dict['MV_GA'], **major_dict['MV']}
    for key in major_dict.keys():
        if len(major_dict[key].keys()) == 0:
            continue
        all_rows.append(ModelsUtils.classify_ensemble(key, major_dict[key], all_features, all_labels, test_indices, nca_selected_idx))

    ##### ============= RUN Stacking ============= #####
    all_rows.append(ModelsUtils.classify_stacking(f'STACKING', major_dict['MV'], all_features, all_labels, test_indices, nca_selected_idx))
    
    for key in major_dict.keys():
        if len(major_dict[key].keys()) == 0:
            continue
        all_rows.append(ModelsUtils.our_classify_stacking(f'OSTACKING {key.strip("MV_")}', major_dict[key], all_features, all_labels, test_indices, nca_selected_idx))
    
    
    ##### ============= RUN CV MV & Stacking ============= #####
    for key in major_dict.keys():
        if len(major_dict[key].keys()) == 0:
            continue
        all_rows.append(ModelsUtils.mv_cv(key + ' CV', major_dict[key], all_features, all_labels, test_indices, nca_selected_idx))
    
    for key in major_dict.keys():
        if len(major_dict[key].keys()) == 0:
            continue
        all_rows.append(ModelsUtils.stacking_cv(f'OSTACKING {key.strip("MV_")} CV', major_dict[key], all_features, all_labels, test_indices, nca_selected_idx))
        
            
    #### ---------- Priniting table ---------- ####
    print('')
    table_headers = ["Classifier", "Success Rate", "Classifier Prediction", "Test Labels", "Sub Labels"]
    print(tabulate(all_rows, headers=table_headers))

    all_rows.insert(0, table_headers[:2])
    Utils.create_sub_folder(folder_name=args_dict['new_folder'])
    np.savetxt(f'class_results/{args_dict["new_folder"]}/{folder_dict["name"]}_{folder_dict["date"]}_{folder_dict["num"]}.csv', np.array(all_rows, dtype=object), delimiter=',', fmt='%s')
    chosen_headers = list(chosen_indices.keys())
    chosen_vals = list(chosen_indices.values())
    chosen_table = np.array([chosen_headers, chosen_vals])
    np.savetxt(f'class_results/{args_dict["new_folder"]}/chosen_features_{folder_dict["name"]}_{folder_dict["date"]}_{folder_dict["num"]}.csv', np.array(chosen_table, dtype=object), delimiter=',', fmt='%s')


if __name__ == '__main__':
    args_dict = Parsers.parse_cmdl_offline()
    if args_dict['paths'] is None:
        classify(args_dict)
    else:
        if args_dict['unify'] is None:
            paths = Utils.get_paths(paths_file=args_dict['paths'])
            for path in paths:
                args_dict['folder'] = path
                print(f'running now classify on path: {path}')
                classify(args_dict)

        else:
            paths = Utils.get_paths(paths_file=args_dict['paths'],unify=args_dict['unify'])
            for path in paths:
                args_dict['folder'] = path[0]
                print(f'first path: {path[0]}')
                if len(path) == 2:
                    args_dict['folder2'] = path[1]
                    print(f'second path: {path[1]}')                    
                classify(args_dict)