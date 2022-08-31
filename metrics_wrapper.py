import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt

classes_map = {'idle': 1, 'left': 2, 'right': 3}

def get_paths():
    paths = open('paths_TK.txt', 'r')
    list_of_paths = [line.strip() for line in paths.readlines()]
    return list_of_paths


def get_all_labels_features_from_folder(folder):
    # need to take the features matrix and trainingvec
    all_features = sio.loadmat(folder + '\AllDataInFeatures.mat')['AllDataInFeatures']
    all_labels = sio.loadmat(folder + '\\trainingVec.mat')['trainingVec'].T

    features_lables = np.concatenate((all_labels, all_features), axis=1)
    df = pd.DataFrame(features_lables,columns=['Class'] + [f'F{i}' for i in range(1, 224)])
    return df

def get_metrics(label, feature, df):
    sample_label = df[df['Class'] == classes_map[label]][feature]
    return np.mean(sample_label), np.var(sample_label)


if __name__ == "__main__":
    feature = 2
    paths = get_paths()
    metrics_right = []
    metrics_left = []
    for path in paths:
        df = get_all_labels_features_from_folder(path)
        
        #get mean and var for each class
        mean_right, var_right = get_metrics('right', f'F{feature}', df)
        mean_left, var_left = get_metrics('left', f'F{feature}', df)
        
        #save values to list
        metrics_right.append((mean_right, var_right))
        metrics_left.append((mean_left, var_left))
    x_axis = [name.split('\\')[-3] + '-' +  name.split('\\')[-1].strip('Sub').split('00')[-1] for name in paths]
    plt.plot(x_axis, [metric[0] for metric in metrics_right])
    plt.plot(x_axis, [metric[0] for metric in metrics_left])
    plt.legend(labels=['Right', 'Left'])
    plt.title(f'Means of feature {feature} over time', size=21)
    plt.xlabel('Recording ID', size=18)
    plt.ylabel(f'Mean value of feature {feature}', size=18)
    plt.savefig('Means', dpi=600)
    plt.show()

    plt.plot(x_axis, [metric[1] for metric in metrics_right])
    plt.plot(x_axis, [metric[1] for metric in metrics_left])
    plt.legend(labels=['Right', 'Left'])
    plt.title(f'Variance of feature {feature} over time', size=21)
    plt.xlabel('Recording ID', size=18)
    plt.ylabel(f'Variance value of feature {feature}', size=18)
    plt.savefig("Vars", dpi=600)
    plt.show()
