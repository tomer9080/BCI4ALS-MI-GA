import os
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# rootdir = "C:\\Users\\Latzres\\Desktop\\project\\BCI-Matlab-Code\\class_results"
# scores_path = "C:\\Users\\Latzres\\Desktop\\project\\BCI-Matlab-Code\\class_scores"

def create_path(base_parts):
    path = ""
    for part in base_parts:
        path += f"{part}\\"
    return path[:-1]

def analyze_csv(scores_path):
    for subdir, dirs, files_top in os.walk(scores_path):
        subdir_list = Path(subdir).parts
        subdir_name = subdir_list[len(subdir_list)-1]
        if(subdir != scores_path and  subdir_name != 'None'):
            print("starting analysis for:")
            print(subdir_name)
            print()
            for x, y, file in os.walk(subdir):
                #load csv
                file_str = file[0]
                df = pd.read_csv(os.path.join(subdir, file_str), header=0)
                print(f'here {os.path.join(subdir, file_str)}')
                print(df.head())
                #extract headers
                headers = list(df.columns.values)
                del headers[0]
                print(headers)

                #extract recordings
                recordings = df['Recording']
                recordings = [i.replace('.csv', '') for i in recordings]

                #create plot for each header
                means = []
                means_headers = []
                for header in headers:
                    values_list = np.array(df[header])
                    mean_header = np.mean(values_list)
                    # plt.clf()
                    # fig = plt.figure(figsize=(6.4, 9.4))
                    # plt.plot(recordings, values_list)
                    # plt.title(f'{subdir_name} \n {header} classification \n mean value = {mean_header}')
                    # plt.xlabel('Recording ID')
                    # plt.ylabel('classification percentage')
                    # plt.xticks(fontsize=6, rotation=90)

                    # print(f'{header} mean = {mean_header}\n')
                    means.append(format(float(mean_header), '.2f'))
                    means_headers.append(header)
                    # plt.savefig(f'{subdir}\\{header}', dpi=600)
                means_table = [means_headers, means]
                np.savetxt(os.path.join(subdir, 'models_avg_scores.csv'), np.array(means_table, dtype=object), delimiter=',', fmt='%s')



def csv_scores(rootdir, base_path):
    for subdir, dirs, files_r in os.walk(rootdir):
        if(subdir != rootdir):
            class_flag = True
            all_rows = []
            for x, y, files_s in os.walk(subdir):
                for file in files_s: 
                    print(f'{subdir}\\{file}')
                    if 'chosen' in file:
                        continue
                    df = pd.read_csv(os.path.join(subdir, file), usecols=[0, 1], header=0)
                    #get classifier stats
                    df['Success Rate'] = df['Success Rate'].apply(lambda x: format(float(x), '.2f'))
                    success_rate = df['Success Rate'].tolist()
                    success_rate.insert(0, file)
                    all_rows.append(success_rate)

                    #get all classifiers in list(needed just once)
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
                full_path = os.path.join(base_path, 'class_scores', subdir_name[-1])
                os.makedirs(full_path, exist_ok=True)
                np.savetxt(os.path.join(full_path, 'classifiers_scores.csv'), all_rows, delimiter=',', fmt='%s')



if __name__ == '__main__':
    #create base path
    rootdir = sys.argv[1]
    rootdir_parts = Path(rootdir).parts
    base_parts = rootdir_parts[:(len(rootdir_parts)-1)]
    base_path = create_path(base_parts)
    print(base_path)

    csv_scores(rootdir, base_path)

    scores_path = "class_scores"
    analyze_csv(scores_path)