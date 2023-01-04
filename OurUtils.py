import pickle
import os
import scipy.io as sio
import numpy as np
import pandas as pd

features_names_list = ['BP_15.5_18.5', 'BP_8_10.5', 'BP_10_15.5', 'BP_17.5_20.5', 'BP_12.5_30', 'RTP', 'SPEC_MOM', 'SPEC_EDGE', 'SPEC_ENT', 'SLOPE', 'INTERCEPT', 'MEAN_FREQ', 'OCC_BAND', 'POWER_BAND', 'WLT_ENT', 'KURT', 'SKEW', 'VAR', 'STD', 'LOG_ENE_ENT', 'BETA_ALPHA_RATIO', 'BP_THETA']
headers = np.array(['CSP1', 'CSP2', 'CSP3'] + [f'E{i}_{feature}' for i in range(1,12) for feature in features_names_list])


def save_to_pickle(model, path, args: dict={}):
    if args.get('save_models', False):
        subdir = get_subdir(args['folder'])
        create_sub_folder_models(subdir)
        pickle.dump(model, file=open(f'models\{subdir}\{path}', 'wb'))


def load_from_pickle(path):
    model = pickle.load(open(path, 'rb'))
    return model


def get_subdir(path) -> str:
    return get_subdir_from_full_path_dict(get_dict_for_folder_from_path(path))


def get_subdir_from_full_path_dict(path_dict: dict) -> str:
    return path_dict['name'] + '_' + path_dict['date'] + '_' + str(path_dict['num'])


def get_dict_for_folder_from_path(path) -> dict:
    list_of_path = path.split('\\')
    return {"name": list_of_path[-2], "date": list_of_path[-3], "num": int(list_of_path[-1][-2:])}


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


def get_all_features(recordingFolder, recordingFolder_2="None", unify=False):
    all_features = sio.loadmat(recordingFolder + '\AllDataInFeatures.mat')['AllDataInFeatures']
    all_labels = sio.loadmat(recordingFolder + '\\trainingVec.mat')['trainingVec'].ravel()
    test_indices = sio.loadmat(recordingFolder + '\\testIdx.mat')['testIdx'].ravel()
    nca_selected_idx = sio.loadmat(recordingFolder + '\\SelectedIdx.mat')['SelectedIdx'].ravel() - 1 

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

def create_sub_folder(folder_name):
    try:
        os.mkdir(f'class_results\{folder_name}')
    except FileExistsError:
        print(f"{folder_name} Already exists, moving on...")    


def create_sub_folder_models(folder_name):
    try:
        os.mkdir(f'models\{folder_name}')
    except FileExistsError:
        print(f"{folder_name} Already exists, moving on...")    


def create_sub_folder_for_ga_features(folder_name):
    try:
        os.mkdir(f'ga_features\{folder_name}')
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
