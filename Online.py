"""
Online -  this module is a script that can run an asked model, given its weights, and the features selected,
and run and classify the wanted recodings given by a file / list.
TODO: Make adjustments using quarter of the samples (fine tuning for online learning)
TODO: be able to use any classifier.
TODO: use pickle to load the classifier of the desired run
TODO: use parser to tell 
"""
import OurUtils as Utils
import numpy as np
import ModelsUtils
import ModelsParams
import pandas as pd
import os

from pathlib import Path
from tabulate import tabulate
from Parsers import parse_cmdl_online

def build_ga_online_models(features, labels):
    # iterate over all the models
    ga_online_models = ModelsParams.build_models()
    for model in ga_online_models:
        reduced_features, mask = ModelsUtils.reduce_ga_search_space(features, model.get('name'))
        model['model'].fit(reduced_features, labels)
        model['indices'] = mask
    return ga_online_models


def run_online_ga(args_dict: dict):
    # build models
    features, labels, _, _ = Utils.get_all_features(args_dict['offline'])
    ga_online_models = build_ga_online_models(features, labels)
    
    # now get new features for online classification
    features, labels, _, _ = Utils.get_all_features(args_dict['online'])
    rows = []
    for model in ga_online_models:
        rows.append(ModelsUtils.classify_online_model(model['model'], model['name'], model['indices'], features, labels))
    
    print('')
    table_headers = ["Classifier", "Success Rate", "Classifier Prediction", "Test Labels", "Sub Labels"]
    print(tabulate(rows, headers=table_headers, maxcolwidths=[300] * 5))


def run_online(args_dict: dict):
    offline_models_path = os.path.join('models', args_dict['name'], Utils.get_subdir(args_dict['offline']))
    online_models_path = f'models/{Utils.get_subdir(args_dict["online"])}'
    all_features, all_labels, _, _ = Utils.get_all_features(args_dict['online'])
    rows = []
    for subdir, _, _ in os.walk('models'):
        if subdir != offline_models_path:
            continue
        for _, _, files_s in os.walk(subdir):
            for file in files_s:
                offline_model = Utils.load_from_pickle(f'{subdir}\\{file}')
                features_df = Utils.get_by_name_and_recording_selected_features_df(args_dict['name'], args_dict['online'])
                model_name = (Path(file).parts)[-1].replace('_object.pkl', '')
                if 'STA' in file:
                    sta_row = ModelsUtils.classify_online_model(offline_model, model_name, features_df[1], all_features, all_labels)
                    rows.append(sta_row)
                else:
                    matlab_row = ModelsUtils.classify_online_model(offline_model, model_name, features_df[0], all_features, all_labels)
                    rows.append(matlab_row)
                
    ### ------ Print Results ------ ###
    print('')
    table_headers = ["Classifier", "Success Rate", "Classifier Prediction", "Test Labels", "Sub Labels"]
    print(tabulate(rows, headers=table_headers, maxcolwidths=[300] * 5))


if __name__ == "__main__":
    args_dict = parse_cmdl_online()
    import sys
    print(sys.argv)
    # run_online(args_dict)
    run_online_ga(args_dict)