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

def plot_class_per_record_to_remove():
    file_name_1 = 'classifiers_scores.csv'
    file_name_2 = 'models_avg_scores.csv'
    root_dir = os.path.join('class_scores')

    models_scores_per_blitz = {
        'LDA GA CV': [],
        'KNN5 GA CV': [],
        'KNN7 GA CV': [],
        'SVC GA CV': [],
        'NB GA CV': [],
        'LR GA CV': []
    }
    models_avg_per_blitz = {
        'LDA GA CV': [],
        'KNN5 GA CV': [],
        'KNN7 GA CV': [],
        'SVC GA CV': [],
        'NB GA CV': [],
        'LR GA CV': []
    }
    # Traverse through the directory tree
    for root, directories, files in os.walk(root_dir):
        for file in files:
            # Check if the file is a CSV
            csv_path = os.path.join(root, file)
            if file.endswith(file_name_2):
                                # Read the CSV file into a DataFrame
                df_1 = pd.read_csv(csv_path)

                # Filter columns containing 'GA CV' in column name
                filtered_columns = [col for col in df_1.columns if 'GA CV' in col]
                df_1 = df_1[filtered_columns]

                for key in models_scores_per_blitz.keys():
                    models_avg_per_blitz[key].append(df_1[key].to_list()[0])
            if file.endswith(file_name_1):

                # Read the CSV file into a DataFrame
                df = pd.read_csv(csv_path)

                # Filter columns containing 'GA CV' in column name
                filtered_columns = [col for col in df.columns if 'GA CV' in col] + ['Recording']
                df = df[filtered_columns]
                recordings = df['Recording'].to_list()
                for key in models_scores_per_blitz.keys():
                    models_scores_per_blitz[key].append(df[key].to_list())


    plot_error_bar_to_remove(models_scores_per_blitz, recordings)
    plot_error_bar_per_model(models_avg_per_blitz)


def plot_error_bar_per_model(models_avg_per_blitz: dict):
    x = models_avg_per_blitz.keys()
    y = [np.mean(item) for item in models_avg_per_blitz.values()]
    y_err = [np.std(item) for item in models_avg_per_blitz.values()]
    plt.figure(figsize=(8, 6))
    plt.errorbar(x, y, y_err, fmt='s', capsize=6)
    # Add labels and title
    plt.xlabel('Model')
    plt.xticks(rotation=30)
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
    plt.ylabel('Values')
    plt.title(f'Error Bars vs Recordings')
    plt.legend()

    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    plot_class_per_record_to_remove()