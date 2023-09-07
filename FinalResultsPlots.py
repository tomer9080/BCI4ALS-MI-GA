# TODO: 0. Remove noisy recordings.
# TODO: 1. For each thresh - plot for all of the classifiers errorbar of the CV scores.
# TODO: 2. Plot bar / errorbar including all threshold and all classifiers.
# TODO: 3. 

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import OurUtils as Utils
from pathlib import Path
from OurUtils import from_feature_name_to_index
import sys

def plot_class_per_record_to_remove(thresh=f"thresh_0", knn_exist=True):
    file_name_1 = 'classifiers_scores.csv'
    file_name_2 = 'models_avg_scores.csv'
    root_dir = os.path.join('class_scores') if knn_exist else os.path.join('class_scores_no_knn')

    models_scores_per_blitz = {
        'LDA GA CV': [],
        'KNN5 GA CV': [],
        'KNN7 GA CV': [],
        'SVC GA CV': [],
        'NB GA CV': [],
        'LR GA CV': [],
        'MV_GA CV': [],
        'OSTACKING GA CV': []
    }

    models_avg_per_blitz = {
        'LDA CV': [],
        'KNN5 CV': [],
        'KNN7 CV': [],
        'SVC CV': [],
        'NB CV': [],
        'LR CV': [],
        'LDA GA CV': [],
        'KNN5 GA CV': [],
        'KNN7 GA CV': [],
        'SVC GA CV': [],
        'NB GA CV': [],
        'LR GA CV': [],
        'OSTACKING  CV': [],
        'OSTACKING ALL CV': [],
        'OSTACKING GA CV': [],
        'MV CV': [],
        'MV_GA CV': [],
        'MV_ALL CV': [],
    }
    

    models_scores_per_blitz = models_scores_per_blitz if knn_exist else {key: val for key, val in models_scores_per_blitz.items() if 'KNN' not in key}
    models_avg_per_blitz = models_avg_per_blitz if knn_exist else {key: val for key, val in models_avg_per_blitz.items() if 'KNN' not in key}

    # Traverse through the directory tree
    for root, directories, files in os.walk(root_dir):
        for file in files:
            # Check if the file is a CSV
            csv_path = os.path.join(root, file)
            if file.endswith(file_name_2) and thresh in csv_path:
                # Read the CSV file into a DataFrame
                df = pd.read_csv(csv_path)

                # Filter columns containing 'GA CV' in column name
                filtered_columns = [col for col in df.columns if 'CV' in col]
                filtered_columns_non_cv = [col for col in df.columns if 'CV' not in col]
                df_1 = df[filtered_columns]
                df_2 = df[filtered_columns_non_cv]

                for key in models_avg_per_blitz.keys():
                    models_avg_per_blitz[key].append(df_1[key].to_list()[0])
                
                
            if file.endswith(file_name_1) and thresh in csv_path:

                # Read the CSV file into a DataFrame
                df = pd.read_csv(csv_path)

                # Filter columns containing 'GA CV' in column name
                filtered_columns = [col for col in df.columns if 'GA CV' in col] + ['Recording']
                df = df[filtered_columns]
                recordings = df['Recording'].to_list()
                for key in models_scores_per_blitz.keys():
                    models_scores_per_blitz[key].append(df[key].to_list())


    plot_error_bar_to_remove(models_scores_per_blitz, recordings)
    return plot_error_bar_per_model_cv(models_avg_per_blitz, thresh=int(thresh.strip('thresh_')), knn_exist=knn_exist)
    # plot_error_bar_per_model_non_cv(models_avg_per_blitz_non_cv)


def plot_error_bar_per_model_cv(models_avg_per_blitz: dict, thresh=0, knn_exist=True):
    x = ['LDA', 'KNN5', 'KNN7', 'SVC', 'NB', 'LR', 'STACKING', 'MV'] if knn_exist else ['LDA', 'SVC', 'NB', 'LR', 'STACKING', 'MV']
    num_blitz = len(models_avg_per_blitz['LDA CV'])
    y_1 = [np.mean(item) * 100 for key, item in models_avg_per_blitz.items() if 'CV' in key and 'GA' not in key and 'ALL' not in key]
    y_1_err = [np.std(item) * 100 for key, item in models_avg_per_blitz.items() if 'CV' in key and 'GA' not in key and 'ALL' not in key]
    y_2 = [np.mean(item) * 100 for key, item in models_avg_per_blitz.items() if 'GA CV' in key]
    y_2_err = [np.std(item) * 100 for key, item in models_avg_per_blitz.items() if 'GA CV' in key]
    y_3 = [np.mean(item) * 100 for key, item in models_avg_per_blitz.items() if 'ALL CV' in key]
    y_3_err = [np.std(item) * 100 for key, item in models_avg_per_blitz.items() if 'ALL CV' in key]

    plt.figure(figsize=(8, 6))
    plt.errorbar(x, y_1, y_1_err, fmt='s', capsize=6, zorder=1, label='Base (NCA)')
    plt.errorbar(x, y_2, y_2_err, fmt='s', capsize=6, zorder=1, label='GA')
    plt.errorbar(x[-2:], y_3, y_3_err, fmt='s', capsize=6, zorder=1, label='Base + GA')
    plt.legend()
    plt.legend().set_title('Feature Selection Method')
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Success Rate (%)')
    plt.title(f'CV Success Rate vs Model for each feature selection method \n Averaged on {num_blitz} runs, GA threshold {thresh}')

    plt.grid(True)
    plt.show()
    return [y_2[-1], y_2_err[-1]]

def plot_error_bar_per_model_non_cv(models_avg_per_blitz: dict):
    x = ['LDA', 'KNN5', 'KNN7', 'SVC', 'NB', 'LR', 'MV', 'STACKING']
    y_1 = [np.mean(item) for key, item in models_avg_per_blitz.items() if 'GA' not in key and 'ALL' not in key]
    y_1_err = [np.var(item) for key, item in models_avg_per_blitz.items() if 'GA' not in key and 'ALL' not in key]
    y_1_med = [np.median(item) for key, item in models_avg_per_blitz.items() if 'GA' not in key and 'ALL' not in key]
    y_2 = [np.mean(item) for key, item in models_avg_per_blitz.items() if 'GA' in key]
    y_2_err = [np.var(item) for key, item in models_avg_per_blitz.items() if 'GA' in key]
    y_2_med = [np.median(item) for key, item in models_avg_per_blitz.items() if 'GA' in key]
    y_3 = [np.mean(item) for key, item in models_avg_per_blitz.items() if 'ALL' in key]
    y_3_err = [np.var(item) for key, item in models_avg_per_blitz.items() if 'ALL' in key]
    y_3_med = [np.median(item) for key, item in models_avg_per_blitz.items() if 'ALL' in key]


    plt.figure(figsize=(8, 6))
    plt.errorbar(x, y_1, y_1_err, fmt='s', capsize=6)
    plt.scatter(x, y_1_med, color='red', marker='x', label='Median')
    plt.errorbar(x, y_2, y_2_err, fmt='s', capsize=6)
    plt.scatter(x, y_2_med, color='red', marker='x', label='Median')
    plt.errorbar(x[-2:], y_3, y_3_err, fmt='s', capsize=6)
    plt.scatter(x[-2:], y_3_med, color='red', marker='x', label='Median')
    plt.legend(['Base', 'GA', 'Base + GA'])
    # Add labels and title
    plt.xlabel('Model')
    plt.xticks(rotation=15)
    plt.ylabel('Success Rate')
    plt.title(f'Success Rate vs Model')

    plt.grid(True)
    plt.show()


def plot_error_bar_to_remove(models_scores_per_blitz: dict, recordings):
    plt.figure(figsize=(8, 6))
    for key in models_scores_per_blitz.keys():
        # Create a scatter plot
        tmp_np = np.array(models_scores_per_blitz[key], dtype=float)
        means = np.mean(tmp_np, axis=0)
        vars = np.var(tmp_np, axis=0)

        plt.errorbar(recordings, means, vars, fmt='o', label=f'{key}')

    # Add labels and title
    plt.xlabel('Recordings')
    plt.xticks(rotation=30)
    plt.ylabel('Success Rate Values')
    plt.title(f'Error Bars vs Recordings')
    plt.legend()

    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    mvga_x = ['MVGA, T=0', 'MVGA, T=50', 'MVGA, T=80', 'MVGA No KNN, T=0', 'MVGA No KNN, T=50', 'MVGA No KNN, T=80']
    mvga_scores = []
    mvga_scores.append(plot_class_per_record_to_remove('thresh_0'))
    mvga_scores.append(plot_class_per_record_to_remove('thresh_50'))
    mvga_scores.append(plot_class_per_record_to_remove('thresh_80'))
    mvga_scores.append(plot_class_per_record_to_remove('thresh_0', knn_exist=False))
    mvga_scores.append(plot_class_per_record_to_remove('thresh_50', knn_exist=False))
    mvga_scores.append(plot_class_per_record_to_remove('thresh_80', knn_exist=False))
    
    mvga_scores = np.array(mvga_scores, dtype=float)

    plt.figure(figsize=(8, 6))
    plt.errorbar(mvga_x, mvga_scores[:, 0], mvga_scores[:, 1], fmt='s', capsize=6)
    # Add labels and title
    plt.xlabel('Model Configuration (T is for Threshold)')
    plt.xticks(rotation=30)
    plt.ylabel('Success Rate (%)')
    plt.ylim((65, 72))
    plt.title(f'CV Success Rate vs Model for each MVGA classifier configuration \n Threshold is on the number of times each feature was chosen in previous runs.')

    plt.grid(True)
    plt.show()
