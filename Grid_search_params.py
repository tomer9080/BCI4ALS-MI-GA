from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB as NB
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier



def build_gs_models(train_features_nca, test_features_nca, labels_train_nca, labels_test_nca, our_features_indices, train_features_stats, test_features_stats, labels_train_stats, labels_test_stats):

    ## LogisticRegression params
    LR_model = LogisticRegression(max_iter=1000)
    LR_solvers = ['newton-cg', 'lbfgs', 'liblinear']
    LR_penalty = ['l2']
    LR_c_values = [100, 10, 1.0, 0.1, 0.01]
    # define grid search
    LR_grid = dict(solver=LR_solvers,penalty=LR_penalty,C=LR_c_values)
    LR_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    LR_grid_search = GridSearchCV(estimator=LR_model, param_grid=LR_grid, n_jobs=-1, cv=LR_cv, scoring='accuracy',error_score=0)
    # grid_result = grid_search.fit(X, y)

    ## Ridge Classifier params
    ridge_model = RidgeClassifier(max_iter=1000)
    ridge_alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # define grid search
    ridge_grid = dict(alpha=ridge_alpha)
    ridge_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    ridge_grid_search = GridSearchCV(estimator=ridge_model, param_grid=ridge_grid, n_jobs=-1, cv=ridge_cv, scoring='accuracy',error_score=0)

    ## KNN params
    KNN_model = KNN()
    n_neighbors = range(1, 21, 2)
    weights = ['uniform', 'distance']
    KNN_metric = ['euclidean', 'manhattan', 'minkowski']
    # define grid search
    KNN_grid = dict(n_neighbors=n_neighbors,weights=weights,metric=KNN_metric)
    KNN_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    KNN_grid_search = GridSearchCV(estimator=KNN_model, param_grid=KNN_grid, n_jobs=-1, cv=KNN_cv, scoring='accuracy',error_score=0)

    ## SVC params
    SVC_model = SVC()
    SVC_kernel = ['poly', 'rbf', 'sigmoid']
    C = [50, 10, 1.0, 0.1, 0.01]
    gamma = ['scale']
    # define grid search
    SVC_grid = dict(kernel=SVC_kernel,C=C,gamma=gamma)
    SVC_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    SVC_grid_search = GridSearchCV(estimator=SVC_model, param_grid=SVC_grid, n_jobs=-1, cv=SVC_cv, scoring='accuracy',error_score=0)

    ## Bagged Decision Trees params
    bag_model = BaggingClassifier()
    bag_n_estimators = [10, 100, 1000]
    # define grid search
    bag_grid = dict(n_estimators=bag_n_estimators)
    bag_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    bag_grid_search = GridSearchCV(estimator=bag_model, param_grid=bag_grid, n_jobs=-1, cv=bag_cv, scoring='accuracy',error_score=0)

    ## RF params
    RF_model = RF()
    RF_n_estimators = [10, 100, 1000]
    RF_max_features = ['sqrt', 'log2']
    # define grid search
    RF_grid = dict(n_estimators=RF_n_estimators,max_features=RF_max_features)
    RF_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    RF_grid_search = GridSearchCV(estimator=RF_model, param_grid=RF_grid, n_jobs=-1, cv=RF_cv, scoring='accuracy',error_score=0)

    ## gradient boost params
    gb_model = GradientBoostingClassifier()
    gb_n_estimators = [10, 100, 1000]
    gb_learning_rate = [0.001, 0.01, 0.1]
    gb_subsample = [0.5, 0.7, 1.0]
    gb_max_depth = [3, 7, 9]
    # define grid search
    gb_grid = dict(learning_rate=gb_learning_rate, n_estimators=gb_n_estimators, subsample=gb_subsample, max_depth=gb_max_depth)
    gb_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    gb_grid_search = GridSearchCV(estimator=gb_model, param_grid=gb_grid, n_jobs=-1, cv=gb_cv, scoring='accuracy',error_score=0)


    gs_models = [

        {'name': 'GS LogisticRegression', 'model': LR_model, 'cv': True, 'grid': LR_grid_search},
        {'name': 'GS LogisticRegression NCA', 'model': LR_model, 'cv': True, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca, 'grid': LR_grid_search},
        {'name': 'GS LogisticRegression STA', 'model': LR_model, 'cv': True, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats, 'grid': LR_grid_search},

        {'name': 'GS Ridge', 'model': ridge_model, 'cv': True, 'grid': ridge_grid_search},
        {'name': 'GS Ridge NCA', 'model': ridge_model, 'cv': True, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca, 'grid': ridge_grid_search},
        {'name': 'GS Ridge STA', 'model': ridge_model, 'cv': True, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats, 'grid': ridge_grid_search},
 
        {'name': 'GS KNN', 'model': KNN_model, 'cv': True, 'grid': KNN_grid_search},
        {'name': 'GS KNN NCA', 'model': KNN_model, 'cv': True, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca, 'grid': KNN_grid_search},
        {'name': 'GS KNN STA', 'model': KNN_model, 'cv': True, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats, 'grid': KNN_grid_search},
 
        {'name': 'GS SVC', 'model': SVC_model, 'cv': True, 'grid': SVC_grid_search},
        {'name': 'GS SVC NCA', 'model': SVC_model, 'cv': True, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca, 'grid': SVC_grid_search},
        {'name': 'GS SVC STA', 'model': SVC_model, 'cv': True, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats, 'grid': SVC_grid_search},
 
        {'name': 'GS Bag', 'model': bag_model, 'cv': True, 'grid': bag_grid_search},
        {'name': 'GS Bag NCA', 'model': bag_model, 'cv': True, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca, 'grid': bag_grid_search},
        {'name': 'GS Bag STA', 'model': bag_model, 'cv': True, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats, 'grid': bag_grid_search},
 
        {'name': 'GS RF', 'model': RF_model, 'cv': True, 'grid': RF_grid_search},
        {'name': 'GS RF NCA', 'model': RF_model, 'cv': True, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca, 'grid': RF_grid_search},
        {'name': 'GS RF STA', 'model': RF_model, 'cv': True, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats, 'grid': RF_grid_search},
 
        {'name': 'GS Gradient Boost', 'model': gb_model, 'cv': True, 'grid': gb_grid_search},
        {'name': 'GS Gradient Boost NCA', 'model': gb_model, 'cv': True, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca, 'grid': gb_grid_search},
        {'name': 'GS Gradient Boost STA', 'model': gb_model, 'cv': True, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats, 'grid': gb_grid_search}
 
    ]

    return gs_models