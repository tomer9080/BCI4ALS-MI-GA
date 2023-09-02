import numpy as np
import ModelsUtils
import ModelsParams
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


class EA:

    def __init__(self, models: dict, test_indices, nca_indices, labels, features, eta) -> None:
        self.models = models
        self.reg_models = {key: model for key, model in models.items() if 'GA' not in key}
        self.ga_models = {key: model for key, model in models.items() if 'GA' in key}
        self.train_indices = ModelsUtils.get_train_indices(labels=labels, test_indices=test_indices)
        self.test_indices = test_indices
        self.features = features
        self.test_labels = labels[test_indices]
        self.train_labels = labels[self.train_indices]
        self.labels = labels
        self.nca_features_train = (features[self.train_indices,:])[:,nca_indices]
        self.nca_features_test = (features[test_indices,:])[:,nca_indices]
        self.nca_indices = nca_indices
        self.eta = eta
        self.init_weights()

    def init_weights(self):
        self.weights = {key: 1 for key in self.models.keys()} # initializing all weights to 1
    
    def get_ga_features(self, model, name: str, train=True):
        reduced_features, mask = ModelsUtils.reduce_ga_search_space(features=self.features, model_name=name.replace(' GA', ''))
        to_ret = (reduced_features[self.train_indices,:]) if train else (reduced_features[self.test_indices,:])
        return to_ret

    def fit(self):

        prob_matrices = {}
        train_indices = self.train_indices
        
        # Zip all matrices - work line by line to be each prediction.
        # This is "training" the model

        for key, model in self.reg_models.items():
            train_features = self.features[:,self.nca_indices]
            train_features = train_features[train_indices,:]
            prob_matrices[key] = model.predict_proba(train_features)

        for key, model in self.ga_models.items():
            train_features = self.features[train_indices,:]
            prob_matrices[key] = model.predict_proba(train_features)

        for j, i in enumerate(train_indices):
            y_true = self.labels[i]
            for key, matrix in prob_matrices.items():
                prediction = np.argmax(matrix[j,:]) + 1
                if prediction != y_true:
                    self.weights[key] *= (1 - self.eta)
        
        print(self.weights)


    def predict(self, key_name=''):

        final_proba_matrix = np.zeros((len(self.test_indices), 3))

        for key, model in self.reg_models.items():
            features_test = self.features[:,self.nca_indices]
            features_test = features_test[self.test_indices,:]
            final_proba_matrix += (self.weights[key] * model.predict_proba(features_test))

        for key, model in self.ga_models.items():
            features_test = self.features[self.test_indices,:]
            final_proba_matrix += (self.weights[key] * model.predict_proba(features_test))

        prediction = list(np.argmax(final_proba_matrix, axis=1) + 1)
        label_test = self.test_labels
        
        return ModelsUtils.row_to_print(prediction, label_test, key_name)
    
    def run_cv(self, folds):
        kf = KFold(n_splits=folds, shuffle=False)

        ga_features = {}

        # build features dict from GA models
        for key, model in self.ga_models.items():
            ga_features[key] = model.support_

        all_scores = []
        for i, train_index, test_index in enumerate(kf.split(self.features)):

            X_train = self.features[train_index]
            X_test = self.features[test_index]
            y_train = self.labels[train_index]
            y_test = self.labels[test_index]

            # Training each Pseudo GA classifier
            ga_models = ModelsParams.build_models()
            for model in ga_models:
                model['model'].fit(X_train[:, ga_features[model['name']]], y_train)  # fit GA pseudo model
            
            self.ga_models = {f"{model['name']} GA": model['model'] for model in ga_models}

            # == Train ensemble == #
            self.init_weights()
            prob_matrices = {}
            train_indices = self.train_indices
            
            # Zip all matrices - work line by line to be each prediction.
            # This is "training" the model

            for key, model in self.reg_models.items():
                train_features = X_train[:, self.nca_indices]
                prob_matrices[key] = model.predict_proba(train_features)

            for key, model in self.ga_models.items():
                train_features = X_train[:, ga_features[key]]
                prob_matrices[key] = model.predict_proba(train_features)

            for j, i in enumerate(train_indices):
                y_true = self.labels[i]
                for key, matrix in prob_matrices.items():
                    prediction = np.argmax(matrix[j,:]) + 1
                    if prediction != y_true:
                        self.weights[key] *= (1 - self.eta)

            # == Predict == #
            final_proba_matrix = np.zeros((len(self.test_indices), 3))

            for key, model in self.reg_models.items():
                features_test = X_test[:,self.nca_indices]
                final_proba_matrix += (self.weights[key] * model.predict_proba(features_test))

            for key, model in self.ga_models.items():
                features_test = X_test[:, ga_features[key]]
                final_proba_matrix += (self.weights[key] * model.predict_proba(features_test))

            prediction = list(np.argmax(final_proba_matrix, axis=1) + 1)
            label_test = self.test_labels

            # Save prediction
            all_scores.append(accuracy_score(label_test, prediction))

        return np.mean(all_scores)
            


            



