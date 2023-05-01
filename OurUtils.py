import pickle
import sys
import os
import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path

features_names_list = ['BP_15.5_18.5', 'BP_8_10.5', 'BP_10_15.5', 'BP_17.5_20.5', 'BP_12.5_30', 'RTP', 'SPEC_MOM', 'SPEC_EDGE', 'SPEC_ENT', 'SLOPE', 'INTERCEPT', 'MEAN_FREQ', 'OCC_BAND', 'POWER_BAND', 'WLT_ENT', 'KURT', 'SKEW', 'VAR', 'STD', 'LOG_ENE_ENT', 'BETA_ALPHA_RATIO', 'BP_THETA']
headers = np.array(['CSP1', 'CSP2', 'CSP3'] + [f'E{i}_{feature}' for i in range(1,12) for feature in features_names_list])
from_feature_name_to_index = {name: i for i, name in enumerate(headers)}

def save_dict_to_pickle(dict_to_save, name):
    pickle.dump(dict_to_save, file=open(os.path.join('stats', name), 'wb'))

def save_to_pickle(model, path, args: dict={}):
    if args.get('save_models', False):
        subdir = get_subdir(args['folder'])
        create_sub_folder_models(subdir)
        pickle.dump(model, file=open(os.path.join('models', subdir, path), 'wb'))


def load_from_pickle(path):
    model = pickle.load(open(path, 'rb'))
    return model


def get_subdir(path) -> str:
    return get_subdir_from_full_path_dict(get_dict_for_folder_from_path(path))


def get_subdir_from_full_path_dict(path_dict: dict) -> str:
    return path_dict['name'] + '_' + path_dict['date'] + '_' + str(path_dict['num'])


def get_dict_for_folder_from_path(path) -> dict:
    list_of_path = []
    while True:
        path, subdir = os.path.split(path)
        if subdir != "":
            list_of_path.insert(0, subdir)
        else:
            if path != "":
                list_of_path.insert(0, subdir)
            break
    print(list_of_path)
    print(path)
    return {"name": list_of_path[-2], "date": list_of_path[-3], "num": int(list_of_path[-1][-2:])}


def get_matlab_features(recordingFolder, recordingFolder_2, unify):
    # features from matlab neighborhood component analysis - takes 10 best features.
    features_train = sio.loadmat(os.path.join(recordingFolder , 'FeaturesTrainSelected.mat'))['FeaturesTrainSelected']
    label_train = sio.loadmat(os.path.join(recordingFolder , 'LabelTrain.mat'))['LabelTrain'].ravel()

    features_test = sio.loadmat(os.path.join(recordingFolder , 'FeaturesTest.mat'))['FeaturesTest']
    label_test = sio.loadmat(os.path.join(recordingFolder , 'LabelTest.mat'))['LabelTest'].ravel()

    if recordingFolder_2 is not None and unify:
        features_train_2 = sio.loadmat(os.path.join(recordingFolder_2 , 'FeaturesTrainSelected.mat'))['FeaturesTrainSelected']
        label_train_2 = sio.loadmat(os.path.join(recordingFolder_2 , 'LabelTrain.mat'))['LabelTrain'].ravel()

        features_test_2 = sio.loadmat(os.path.join(recordingFolder_2 , 'FeaturesTest.mat'))['FeaturesTest']
        label_test_2 = sio.loadmat(os.path.join(recordingFolder_2 , 'LabelTest.mat'))['LabelTest'].ravel()
        
        features_train = np.concatenate((features_train, features_train_2), axis=0)
        label_train = np.concatenate((label_train, label_train_2), axis=0)
        features_test = np.concatenate((features_test, features_test_2), axis=0)
        label_test = np.concatenate((label_test, label_test_2), axis=0)

    return features_train, label_train, features_test, label_test


def get_all_features(recordingFolder, recordingFolder_2="None", unify=False):
    all_features = sio.loadmat(os.path.join(recordingFolder , 'AllDataInFeatures.mat'))['AllDataInFeatures']
    all_labels = sio.loadmat(os.path.join(recordingFolder , 'trainingVec.mat'))['trainingVec'].ravel()
    test_indices = sio.loadmat(os.path.join(recordingFolder , 'testIdx.mat'))['testIdx'].ravel()
    nca_selected_idx = sio.loadmat(os.path.join(recordingFolder , 'SelectedIdx.mat'))['SelectedIdx'].ravel() - 1 

    if recordingFolder_2 is not None and unify:
        all_features_2 = sio.loadmat(os.path.join(recordingFolder_2 , 'AllDataInFeatures.mat'))['AllDataInFeatures']
        all_labels_2 = sio.loadmat(os.path.join(recordingFolder_2 , 'trainingVec.mat'))['trainingVec'].ravel()
        test_indices_2 = sio.loadmat(os.path.join(recordingFolder_2 , 'testIdx.mat'))['testIdx'].ravel()
        nca_selected_idx_2 = sio.loadmat(os.path.join(recordingFolder_2 , 'SelectedIdx.mat'))['SelectedIdx'].ravel() - 1
        
        all_features = np.concatenate((all_features, all_features_2), axis=0)
        all_labels = np.concatenate((all_labels, all_labels_2), axis=0)
        test_indices = np.concatenate((test_indices, test_indices_2 + len(test_indices)), axis=0)
        
        nca_selected_idx = np.concatenate((nca_selected_idx[:5], nca_selected_idx_2[:5]), axis=0)
        print(f"concatenated headers: {headers[nca_selected_idx]}")
    return all_features, all_labels, test_indices, nca_selected_idx

def create_sub_folder(folder_name):
    try:
        os.mkdir(os.path.join('class_results', folder_name))
    except FileExistsError:
        print(f"{folder_name} Already exists, moving on...")    


def create_sub_folder_models(folder_name):
    try:
        os.mkdir(os.path.join('models', folder_name))
    except FileExistsError:
        print(f"{folder_name} Already exists, moving on...")    


def create_sub_folder_for_ga_features(folder_name):
    try:
        os.mkdir(os.path.join('ga_features', folder_name))
    except FileExistsError:
        print(f"{folder_name} Already exists, moving on...")    


def get_array_from_str(array_str):
    tmp = array_str.replace('[','').replace(']', '')
    tmp = ' '.join(tmp.split())
    tmp = tmp.split()
    return np.array([int(item) for item in tmp])

def get_by_name_and_recording_selected_features_df(name, recording):
    recording_subname = get_subdir(recording)
    df = pd.read_csv(f'class_results\{name}\chosen_features_{recording_subname}.csv', delimiter=',')
    arrays = []
    arrays.append(df['MATLAB'].iloc[0])
    arrays.append(df['STA'].iloc[0])
    features = []
    for array in arrays:
        features.append(get_array_from_str(array))
    return features


def save_best_model_stats(model_name, grid_result):
    stats_file = open(f"GS_stats/{model_name}_stats.txt", 'w')
    stats_file.write(f'{model_name} stats: \n')
    stats_file.write("Best: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))

    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    
    # row = []
    # # for mean, stdev, param in zip(means, stds, params):

    # for i in len(means):
    #     row[i] = [means[i], stds[i], params[i]]

    # table_headers = ["Classifier", "means", "stds", "params"]
    # stats_file.write(tabulate(row, headers=table_headers))
    stats_file.close()

def get_same_indices(nca_features: np.ndarray, all_features: np.ndarray, L: np.ndarray):
    indices = []
    print(nca_features.shape)
    print(L)
    hofhi = (np.linalg.inv(L.T@L))
    print(hofhi.shape)
    muhpal = (L.T @ nca_features)
    print(muhpal.shape)
    nca_features = hofhi @ muhpal
    print(nca_features.shape)
    for i in range(nca_features.shape[1]):
        for j in range(all_features.shape[1]):
            print(all_features[:,j] - nca_features[:,i])
            if np.array_equal(all_features[:,j], nca_features[:,i]):
                indices.append(j)
    return indices


def get_paths(paths_file, is_list=False, unify=False):
    if is_list:
        return [line.strip() for line in paths_file]
    paths = open(paths_file, 'r')
    list_of_paths = [line.strip() for line in paths.readlines()]
    
    if unify:
        unified_list = []
        idx = 0
        while idx <= (len(list_of_paths)-2):
            couples = [list_of_paths[idx]]
            same_date = True
            while(same_date and (idx < (len(list_of_paths)-1))) :
                str1 = Path(list_of_paths[idx]).parts
                str2 = Path(list_of_paths[idx + 1]).parts

                #RL - 7th element of path - 6th in list
                #TK - 3rd element of path - 2nd in list
                print(str1, str2)
                if(str1[2] == str2[2]):
                    couples.append(list_of_paths[idx + 1])
                else:
                    same_date = False
                idx += 1
            unified_list.append(couples)
        return unified_list
    return list_of_paths