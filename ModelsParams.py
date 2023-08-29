from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB as NB
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import GradientBoostingClassifier as GB
from sklearn.ensemble import AdaBoostClassifier as AB


def build_models(our_features_indices=[], train_features_stats=[], test_features_stats=[], labels_train_stats=[], labels_test_stats=[]):
    models = [
        {'name': 'LDA', 'model': LDA(), 'cv': True},
        # {'name': 'LDA STA', 'model': LDA(), 'cv': True, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats},
        {'name': 'KNN-5', 'model': KNN(5), 'cv': True},
        # {'name': 'KNN-5 STA', 'model': KNN(5), 'cv': False, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats},
        {'name': 'KNN-7', 'model': KNN(7), 'cv': True},
        # {'name': 'KNN-7 STA', 'model': KNN(7), 'cv': False, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats},
        {'name': 'SVC', 'model': SVC(probability=True), 'cv': True},
        # {'name': 'SVC STA', 'model': SVC(probability=True), 'cv': True, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats},
        {'name': 'NB', 'model': NB(), 'cv': True},
        {'name': 'LR', 'model': LR(), 'cv': True},
        # {'name': 'NB STA', 'model': NB(), 'cv': False, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats},
        # {'name': 'RF', 'model': RF(criterion='entropy', min_samples_split=3), 'cv': True},
        # {'name': 'RF STA', 'model': RF(criterion='entropy'), 'cv': True, 'indices': our_features_indices, 'ftr': train_features_stats, 'fte': test_features_stats, 'ltr': labels_train_stats, 'lte': labels_test_stats},
    ]

    return models

def build_ga_models(num_max_features):
    ga_models = [        
        {'name': 'LDA', 
         'model': LDA(),
         'cv': 3,
         "scoring": "accuracy",
         "max_features": 20,
         "n_population": 153,
         "cross_prob": 0.73,
         "muta_prob": 0.18,
         "n_gens": 60,
         "caching": True,
         "muta_ind_prob": 0.032,
         "cross_ind_prob": 0.75 },

        {'name': 'KNN5', 
         'model': KNN(5),
         'cv': 3,
         "scoring": "accuracy",
         "max_features": 16,
         "n_population": 160,
         "cross_prob": 0.56,
         "muta_prob": 0.29,
         "n_gens": 40,
         "caching": True,
         "muta_ind_prob": 0.062,
         "cross_ind_prob": 0.41 },
                 
        {'name': 'KNN7', 
         'model': KNN(7),
         'cv': 3,
         "scoring": "accuracy",
         "max_features": 10,
         "n_population": 160,
         "cross_prob": 0.71,
         "muta_prob": 0.19,
         "n_gens": 40,
         "caching": True,
         "muta_ind_prob": 0.038,
         "cross_ind_prob": 0.52 },
         
         {'name': 'SVC', 
         'model': SVC(probability=True),
         'cv': 3,
         "scoring": "accuracy",
         "max_features": 20,
         "n_population": 153,
         "cross_prob": 0.67,
         "muta_prob": 0.3,
         "n_gens": 60,
         "caching": True,
         "muta_ind_prob": 0.055,
         "cross_ind_prob": 0.3 }, 

        {'name': 'NB', 
         'model': NB(),
         'cv': 3,
         "scoring": "accuracy",
         "max_features": 16,
         "n_population": 153,
         "cross_prob": 0.65,
         "muta_prob": 0.32,
         "n_gens": 40,
         "caching": True,
         "muta_ind_prob": 0.09,
         "cross_ind_prob": 0.7 },

          {'name': 'LR', 
          'model': LR(),
          'cv': 3,
          "scoring": "accuracy",
          "max_features": 18,
          "n_population": 153,
          "cross_prob": 0.75,
          "muta_prob": 0.33,
          "n_gens": 60,
          "caching": True,
          "muta_ind_prob": 0.05,
          "cross_ind_prob": 0.7 }, 

        #  {'name': 'RF', 
        #  'model': RF(criterion='entropy', n_estimators=50, min_samples_split=3),
        #  'cv': 3,
        #  "scoring": "accuracy",
        #  "max_features": num_max_features,
        #  "n_population": 153,
        #  "cross_prob": 0.5,
        #  "muta_prob": 0.2,
        #  "n_gens": 30,
        #  "caching": True,
        #  "muta_ind_prob": 0.025,
        #  "cross_ind_prob": 0.8 },

        # {'name': 'DT', 
        #  'model': DT(),
        #  'cv': 3,
        #  "scoring": "accuracy",
        #  "max_features": num_max_features,
        #  "n_population": 153,
        #  "cross_prob": 0.5,
        #  "muta_prob": 0.2,
        #  "n_gens": 30,
        #  "caching": True,
        #  "muta_ind_prob": 0.025,
        #  "cross_ind_prob": 0.8 },

    ]
    
    return ga_models