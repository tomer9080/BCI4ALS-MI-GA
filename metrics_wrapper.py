import numpy as np
import scipy.io as sio
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from tabulate import tabulate
import sys

classes_map = {'idle': 1, 'left': 2, 'right': 3}
features_names_list = ['BP_ALPHA', 'BP_BETA', 'BP_GAMMA', 'BP_DELTA', 'BP_THETA', 'RTP', 'SPEC_MOM', 'SPEC_EDGE', 'SPEC_ENT', 'SLOPE', 'INTERCEPT', 'MEAN_FREQ', 'OCC_BAND', 'POWER_BAND', 'WLT_ENT', 'KURT', 'SKEW', 'VAR', 'STD', 'LOG_ENE_ENT']
headers = ['CSP1', 'CSP2', 'CSP3'] + [f'E{i}_{feature}' for i in range(1,12) for feature in features_names_list]



def get_paths():
    paths = open(sys.argv[1], 'r')
    list_of_paths = [line.strip() for line in paths.readlines()]
    return list_of_paths

def plot_mean_and_variance(paths, metrics_right, metrics_left, feature):
        x_axis = [name.split('\\')[-3] + '-' +  name.split('\\')[-1].strip('Sub').split('00')[-1] for name in paths]
        plt.plot(x_axis, [metric[0] for metric in metrics_right])
        plt.plot(x_axis, [metric[0] for metric in metrics_left])
        plt.legend(labels=['Right', 'Left'])
        plt.title(f'Means of feature {headers[feature]} over time', size=21)
        plt.xlabel('Recording ID', size=18)
        plt.ylabel(f'Mean value of feature {headers[feature]}', size=18)
        plt.savefig('Means', dpi=600)
        plt.show()

        plt.plot(x_axis, [metric[1] for metric in metrics_right])
        plt.plot(x_axis, [metric[1] for metric in metrics_left])
        plt.legend(labels=['Right', 'Left'])
        plt.title(f'Variance of feature {headers[feature]} over time', size=21)
        plt.xlabel('Recording ID', size=18)
        plt.ylabel(f'Variance value of feature {headers[feature]}', size=18)
        plt.savefig("Vars", dpi=600)
        plt.show()

def plot_easy(x, y1, y2=None, xlabel='x axis', ylabel='y axis', title='easy plot', legend=[], save=False):
    plt.plot(x, y1)
    if y2 != None:
        plt.plot(x, y2)
    plt.legend(labels=legend)
    plt.title(title, size=21)
    plt.xlabel(xlabel, size=18)
    plt.ylabel(ylabel, size=18)
    if save:
        plt.savefig(title, dpi=600)
    plt.show()

def reg_plot_easy(x, y1, y2, xlabel='x axis', ylabel='y axis', title='easy plot', legend=[], save=False):
    # regression plot using seaborn
    fig = plt.figure(figsize=(8, 5))
    sns.regplot(x=x, y=y1, color='blue', marker='+')
    sns.regplot(x=x, y=y2, color='magenta', marker='+')
    
    # Legend, title and labels.
    plt.legend(labels=legend)
    plt.title(title, size=24)
    plt.xlabel(xlabel, size=18)
    plt.ylabel(ylabel, size=18)
    plt.show()

def plot_actter_and_lin_reg(x, y ,model):
    x_fit = np.linspace(1, 540, 540).reshape(-1, 1)
    y_fit = model.predict(x_fit)
    plt.scatter(x, y)
    plot_easy(x_fit, y_fit)
    

def get_all_labels_features_from_folder(folder):
    # need to take the features matrix and trainingvec
    all_features = sio.loadmat(folder + '\AllDataInFeatures.mat')['AllDataInFeatures']
    all_labels = sio.loadmat(folder + '\\trainingVec.mat')['trainingVec'].T

    features_lables = np.concatenate((all_labels, all_features), axis=1)
    df = pd.DataFrame(features_lables,columns=['Class'] + headers)
    return df

def get_metrics(label, feature, df):
    sample_label = df[df['Class'] == classes_map[label]][headers[feature]]
    return np.mean(sample_label), np.var(sample_label)


def get_feature_values(label, feature, df):
    feature_list = df[df['Class'] == classes_map[label]][headers[feature]]
    return feature_list

if __name__ == "__main__":
    # feature = 1
    paths = get_paths()
    metrics_right = []
    metrics_left = []

    features_var = []
    features_mean = []

    feature_row = []
    left_values = []
    features_values_dict_r = {key: [] for key in headers}
    features_values_dict_l = {key: [] for key in headers}
    
    x_axis = [i + 1 for i in range(540)]
    x = np.array(x_axis).reshape(-1, 1)

    all_regs_r = []
    all_regs_l = []
    
    for feature in range(len(headers)):  
        tmp_values = []
        for path in paths:
            df = get_all_labels_features_from_folder(path)
            
            #get mean and var for each class
            mean_right, var_right = get_metrics('right', feature, df)
            mean_left, var_left = get_metrics('left', feature, df)

            
            #save values to list
            metrics_right.append((mean_right, var_right))
            metrics_left.append((mean_left, var_left))
            
            #save all trials values to dict
            tmp_values_right = get_feature_values('right', feature, df)
            tmp_values_left = get_feature_values('left', feature, df)
            features_values_dict_r[headers[feature]] += list(tmp_values_right)
            features_values_dict_l[headers[feature]] += list(tmp_values_left)

        avg_mean_right = np.mean([metric[0] for metric in metrics_right])
        avg_mean_left = np.mean([metric[0] for metric in metrics_left])
        
        var_mean_right = np.var([metric[0] for metric in metrics_right])
        var_mean_left = np.var([metric[0] for metric in metrics_left])
        
        avg_var_right = np.mean([metric[1] for metric in metrics_right])
        avg_var_left = np.mean([metric[1] for metric in metrics_left])
        
        var_var_right = np.var([metric[1] for metric in metrics_right])
        var_var_left = np.var([metric[1] for metric in metrics_left])

        y_r = np.array(features_values_dict_r[headers[feature]])
        y_l = np.array(features_values_dict_l[headers[feature]])
        
        reg_r = LinearRegression()
        reg_r.fit(x, y_r)

        reg_l = LinearRegression()
        reg_l.fit(x, y_l)

        all_regs_r.append(reg_r)
        all_regs_l.append(reg_l)
        
        row = [headers[feature], reg_l.score(x, y_l), reg_r.score(x, y_r), avg_mean_left, var_mean_left, avg_mean_right, var_mean_right, avg_var_left, var_var_left, avg_var_right, var_var_right]
        feature_row.append(row)

    reg_plot_easy(x=x_axis, y1=features_values_dict_r['CSP1'], y2=features_values_dict_l['CSP1'], legend=['Right', 'Left'],  xlabel='Num of Trial', ylabel='CSP1 value', title='R vs L values of CSP1')
    plot_easy(x=x_axis, y1=features_values_dict_r['CSP1'], xlabel='Num of Trial', ylabel='CSP1 value', title='CSP1 value over total trials (right)')
    plot_easy(x=x_axis, y1=features_values_dict_l['CSP1'], xlabel='Num of Trial', ylabel='CSP1 value', title='CSP1 value over total trials (left)')

    reg_plot_easy(x=x_axis, y1=features_values_dict_r['CSP2'], y2=features_values_dict_l['CSP2'], legend=['Right', 'Left'],  xlabel='Num of Trial', ylabel='CSP1 value', title='R vs L values of CSP1')
    plot_easy(x=x_axis, y1=features_values_dict_r['CSP2'], xlabel='Num of Trial', ylabel='CSP2 value', title='CSP2 value over total trials (right)')
    plot_easy(x=x_axis, y1=features_values_dict_l['CSP2'], xlabel='Num of Trial', ylabel='CSP2 value', title='CSP2 value over total trials (left)')

    table_headers = ['Feature', 'Score (R^2) Left', 'Score (R^2) Right', 'Mean-Mean left', 'Var-Mean left', 'Mean-Mean right', 'Var-Mean right', 'Mean-Var left', 'Var-Var left', 'Mean-Var right', 'Var-Var right']
    print(tabulate(feature_row, headers=table_headers))

