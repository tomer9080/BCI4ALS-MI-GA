from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB as NB
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import GradientBoostingClassifier as GB


def build_models(train_features_nca, test_features_nca, labels_train_nca, labels_test_nca, our_features_indices, train_features_stats, test_features_stats, labels_train_stats, labels_test_stats):
    models = [
        {'name': 'LDA', 'model': LDA(), 'cv': True},
        {'name': 'LDA NCA', 'model': LDA(), 'cv': True, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca},
        {'name': 'LDA STA', 'model': LDA(), 'cv': True, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats},
        {'name': 'QDA', 'model': QDA(), 'cv': True},
        {'name': 'QDA NCA', 'model': QDA(), 'cv': True, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca},
        {'name': 'QDA STA', 'model': QDA(), 'cv': True, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats},
        {'name': 'KNN-5', 'model': KNN(5), 'cv': False},
        {'name': 'KNN-5 NCA', 'model': KNN(5), 'cv': False, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca},
        {'name': 'KNN-5 STA', 'model': KNN(5), 'cv': False, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats},
        {'name': 'KNN-7', 'model': KNN(7), 'cv': False},
        {'name': 'KNN-7 NCA', 'model': KNN(7), 'cv': False, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca},
        {'name': 'KNN-7 STA', 'model': KNN(7), 'cv': False, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats},
        {'name': 'SVC', 'model': SVC(probability=True), 'cv': True},
        {'name': 'SVC NCA', 'model': SVC(probability=True), 'cv': True, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca},
        {'name': 'SVC STA', 'model': SVC(probability=True), 'cv': True, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats},
        {'name': 'NB', 'model': NB(), 'cv': False},
        {'name': 'NB NCA', 'model': NB(), 'cv': False, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca},
        {'name': 'NB STA', 'model': NB(), 'cv': False, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats},
        {'name': 'RF', 'model': RF(criterion='entropy'), 'cv': True},
        {'name': 'RF NCA', 'model': RF(criterion='entropy'), 'cv': True, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca},
        {'name': 'RF STA', 'model': RF(criterion='entropy'), 'cv': True, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats},
        {'name': 'DT', 'model': DT(), 'cv': True},
        {'name': 'DT NCA', 'model': DT(), 'cv': True, 'ftr': train_features_nca, 'fte': test_features_nca, 'ltr': labels_train_nca, 'lte': labels_test_nca},
        {'name': 'DT STA', 'model': DT(), 'cv': True, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats},
        {'name': 'GB', 'model': GB(), 'cv': True},
    ]

    return models

def build_ga_models(num_max_features):
    ga_models = [        
        {'name': 'LDA', 
         'model': LDA(),
         'cv': 3,
         "scoring": "accuracy",
         "max_features": num_max_features,
         "n_population": 153,
         "cross_prob": 0.5,
         "muta_prob": 0.2,
         "n_gens": 60,
         "caching": True,
         "muta_ind_prob": 0.025,
         "cross_ind_prob": 0.8 },

         {'name': 'QDA', 
         'model': QDA(),
         'cv': 3,
         "scoring": "accuracy",
         "max_features": num_max_features,
         "n_population": 153,
         "cross_prob": 0.5,
         "muta_prob": 0.2,
         "n_gens": 60,
         "caching": True,
         "muta_ind_prob": 0.025,
         "cross_ind_prob": 0.8 },

        {'name': 'KNN5', 
         'model': KNN(5),
         'cv': 3,
         "scoring": "accuracy",
         "max_features": num_max_features,
         "n_population": 153,
         "cross_prob": 0.5,
         "muta_prob": 0.2,
         "n_gens": 30,
         "caching": True,
         "muta_ind_prob": 0.025,
         "cross_ind_prob": 0.8 },
                 
        {'name': 'KNN7', 
         'model': KNN(7),
         'cv': 3,
         "scoring": "accuracy",
         "max_features": num_max_features,
         "n_population": 153,
         "cross_prob": 0.5,
         "muta_prob": 0.2,
         "n_gens": 30,
         "caching": True,
         "muta_ind_prob": 0.025,
         "cross_ind_prob": 0.8 },
         
         {'name': 'SVC', 
         'model': SVC(probability=True),
         'cv': 3,
         "scoring": "accuracy",
         "max_features": num_max_features,
         "n_population": 153,
         "cross_prob": 0.5,
         "muta_prob": 0.2,
         "n_gens": 60,
         "caching": True,
         "muta_ind_prob": 0.025,
         "cross_ind_prob": 0.8 }, 

        {'name': 'NB', 
         'model': NB(),
         'cv': 3,
         "scoring": "accuracy",
         "max_features": num_max_features,
         "n_population": 153,
         "cross_prob": 0.5,
         "muta_prob": 0.2,
         "n_gens": 30,
         "caching": True,
         "muta_ind_prob": 0.025,
         "cross_ind_prob": 0.8 },

         {'name': 'RF', 
         'model': RF(criterion='entropy', n_estimators=50),
         'cv': 3,
         "scoring": "accuracy",
         "max_features": num_max_features,
         "n_population": 153,
         "cross_prob": 0.5,
         "muta_prob": 0.2,
         "n_gens": 30,
         "caching": True,
         "muta_ind_prob": 0.025,
         "cross_ind_prob": 0.8 },

        {'name': 'DT', 
         'model': DT(),
         'cv': 3,
         "scoring": "accuracy",
         "max_features": num_max_features,
         "n_population": 153,
         "cross_prob": 0.5,
         "muta_prob": 0.2,
         "n_gens": 30,
         "caching": True,
         "muta_ind_prob": 0.025,
         "cross_ind_prob": 0.8 },

    ]
    
    return ga_models