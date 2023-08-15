import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
from GAModel import GAModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import cross_val_score
from OurUtils import get_all_features, get_matlab_features, create_study_sub_folder, get_subdir
from ModelsParams import build_models

def get_Xy(path):
    X, y, test_indices, _ = get_all_features(path)
    train_indices = [i for i in range(X.shape[0]) if i not in test_indices]
    return X[train_indices,:], y[train_indices]

def objective(trial, path):
    
    X, y = get_Xy(path)

    classifier_name = trial.suggest_categorical("classifier", ["SVC", "KNN", "LR"])
    if classifier_name == "SVC":
        svc_kernel = trial.suggest_categorical("kernel", ['poly', 'rbf', 'sigmoid'])
        svc_c = trial.suggest_float("C", 1e-5, 100, log=True)
        clf_obj = SVC(gamma="auto", kernel=svc_kernel, C=svc_c)
    elif classifier_name == "KNN":
        knn_ws = trial.suggest_categorical("weights", ['uniform', 'distance'])
        knn_metric = trial.suggest_categorical("metric", ['euclidean', 'manhattan', 'minkowski'])
        knn_K = trial.suggest_categorical("n_neighbors", list(range(3,10,2)))
        clf_obj = KNN(n_neighbors=knn_K, weights=knn_ws, metric=knn_metric)
    elif classifier_name == "LR":
        lr_solvers = trial.suggest_categorical("solver", ['newton-cg', 'lbfgs', 'liblinear'])
        lr_c = trial.suggest_float("C", 1e-5, 100, log=True)
        clf_obj = LR(C=lr_c, solver=lr_solvers)
    
    score = cross_val_score(clf_obj, X, y, n_jobs=-1, cv=5)
    acc = score.mean()
    return acc


def objective_ga(trial, paths, model, model_name):
    
    index = trial.number // 50
    print(index, trial.number)
    path = paths[index]

    X, y = get_Xy(path)
    ga_cv = trial.suggest_int("cv", 3, 5)
    ga_max_features = trial.suggest_int("max_features", 5, 20)
    ga_pop = trial.suggest_int("n_population", 120, 180)
    ga_cross_prob = trial.suggest_float('cross_prob', 0.2, 0.8)
    ga_muta_prob = trial.suggest_float('muta_prob', 0.15, 0.35)
    ga_n_gens = trial.suggest_categorical('n_gens', [40, 50, 60])
    ga_muta_ind_prob = trial.suggest_float('muta_ind_prob', 0.025, 0.1)
    ga_cross_ind_prob = trial.suggest_float('cross_ind_prob', 0.1, 0.8)
    clf_obj = GAModel(model_name, model, cv=ga_cv, max_features=ga_max_features, n_population=ga_pop, cross_prob=ga_cross_prob, muta_prob=ga_muta_prob, n_gens=ga_n_gens, muta_ind_prob=ga_muta_ind_prob, cross_ind_prob=ga_cross_ind_prob)

    clf_obj.fit(X, y)

    # score = clf_obj.get_scores()
    
    acc = clf_obj.score(X, y)
    return acc

def run_optuna(path, num_trials=20):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, path), n_trials=num_trials)
    optuna.visualization.matplotlib.plot_param_importances(study)
    optuna.visualization.matplotlib.plot_contour(study, params=["max_features", "cross_prob"])
    import matplotlib.pyplot as plt
    plt.show()
    print(study.best_trial)

def get_pairs(cols):
    pairs = []
    for i in range(len(cols)):
        for j in range(len(cols) - i):
            if j+i == i:
                continue
            pairs.append([cols[i], cols[j+i]])
    return pairs

def run_optuna_ga(paths):
    models = build_models()
    for model in models:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective_ga(trial, paths, model=model['model'], model_name=model['name']), n_trials=50*len(paths))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig(os.path.join('studies', model['name'], 'param_importances.png'))
        for pair in get_pairs(["max_features", "cross_prob", "muta_prob", "cross_ind_prob", "muta_ind_prob"]):
            plot_optuna_contour(pair, study, model["name"])
        # create_study_sub_folder(save_to)
        study.trials_dataframe().to_csv(os.path.join('studies', model['name'], 'study.csv'))
        print(study.best_trial)

def plot_optuna_contour(pair, study, model):
    optuna.visualization.matplotlib.plot_contour(study, params=pair)
    plt.savefig(os.path.join('studies', model, f'param_contour_{"_".join(pair)}.png'))

if __name__ == "__main__":
    import sys
    print(sys.argv)
    from OurUtils import get_paths
    paths = get_paths('paths/paths_linux.txt')
    try:
        num_trials = sys.argv[1]
    except:
        num_trials = 200
    run_optuna_ga(paths)