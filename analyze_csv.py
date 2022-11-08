import os
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt


# rootdir = "C:\\Users\\Latzres\\Desktop\\project\\BCI-Matlab-Code\\class_results"
base_path = "C:\\Users\\Latzres\\Desktop\\project\\BCI-Matlab-Code"
scores_path = "C:\\Users\\Latzres\\Desktop\\project\\BCI-Matlab-Code\\class_scores"


def analyze_csv():
    for subdir, dirs, files_top in os.walk(scores_path):
        if(subdir != scores_path):
            subdir_name = Path(subdir).parts
            for x, y, file in os.walk(subdir):
                #load csv
                file_str = file[0]
                df = pd.read_csv(f'{subdir}\\{file_str}')
                #extract headers
                headers = list(df.columns.values)
                headers.pop(0)
                print(f'headers:\n {headers}\n')
                #extract recordings
                recordings = df['Recording']
                print(f'recordings = {recordings}\n')
                #create plot for each header
                for header in headers:
                    values_list = np.array(df[header])
                    mean_header = np.mean(values_list)
                    plt.clf()
                    plt.plot(recordings, values_list)
                    plt.title(f'{header} classification \n mean value = {mean_header}')
                    plt.xlabel('Recording ID')
                    plt.ylabel('classification percentage')
                    print(f'{header} mean = {mean_header}\n')
                    plt.savefig(f'{subdir}\\{header}', dpi=600)



def csv_scores(rootdir):
    for subdir, dirs, files_r in os.walk(rootdir):
        if(subdir != rootdir):
            class_flag = True
            all_rows = []
            for x, y, files_s in os.walk(subdir):
                for file in files_s:   
                    df = pd.read_csv(f'{subdir}\\{file}')
                    success_rate = df['Success Rate'].tolist()
                    success_rate.insert(0, file)
                    all_rows.append(success_rate)
                    if(class_flag):
                        classifiers = df['Classifier'].tolist()
                        classifiers.insert(0, 'Recording')
                        class_flag = False
                
                #create full 2d array of classifiers and scores for a specific folder
                all_rows.insert(0,classifiers)
                all_rows = np.array(all_rows, dtype=object)
                # all_rows = np.transpose(all_rows)
                
                #create dirs and save scores csv
                subdir_name = Path(subdir).parts
                full_path = f'{base_path}\\class_scores\\{subdir_name[-1]}'
                os.makedirs(full_path, exist_ok=True)
                np.savetxt(f'{full_path}\\classifiers_scores.csv', all_rows, delimiter=',', fmt='%s')



if __name__ == '__main__':
    csv_scores(sys.argv[1])
    analyze_csv()