import numpy as np
import ModelsUtils
import ModelsParams
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


class Stacking:

    def __init__(self, models: dict, test_indices, nca_indices, labels, features, level1_model) -> None:
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
        self.meta_model = level1_model
    
    def get_ga_features(self, model, name: str, train=True):
        reduced_features, mask = ModelsUtils.reduce_ga_search_space(features=self.features, model_name=name.replace(' GA', ''))
        to_ret = (reduced_features[self.train_indices,:]) if train else (reduced_features[self.test_indices,:])
        return to_ret

    def fit(self):

        # Training the meta_model
        new_dataset_train = []
        new_dataset_test = []
        
        for _, model in self.reg_models.items():
            new_dataset_train.append(model.predict(self.nca_features_train))
            new_dataset_test.append(model.predict(self.nca_features_test))

        for _, model in self.ga_models.items():
            new_dataset_train.append(model.predict(self.features[self.train_indices,:]))
            new_dataset_test.append(model.predict(self.features[self.test_indices,:]))

        new_dataset = np.array(new_dataset_train, dtype=object)
        self.new_dataset = new_dataset.T  # Transposed to make rows to columns.
        print(self.new_dataset.shape)  # should be from shape: (num_train_trials, num_models)

        self.meta_model.fit(self.new_dataset, self.train_labels)

        self.new_dataset_test = np.array(new_dataset_test, dtype=object).T

    def predict(self):
        return self.meta_model.predict(self.new_dataset_test)
    
    def run_cv(self, folds):
        kf = KFold(n_splits=folds, shuffle=False)

        ga_features = {}

        # build features dict from GA models
        for key, model in self.ga_models.items():
            ga_features[key.replace(' GA', '')] = model.support_

        all_scores = []
        for train_index, test_index in kf.split(self.features):
            X_train = self.features[train_index]
            X_test = self.features[test_index]
            y_train = self.labels[train_index]
            y_test = self.labels[test_index]

            # Training the meta_model
            new_dataset_train = []
            new_dataset_test = []
            
            for _, model in self.reg_models.items():
                new_dataset_train.append(model.predict(X_train[:, self.nca_indices]))
                new_dataset_test.append(model.predict(X_test[:, self.nca_indices]))
                # print('Stacking: In normal models')

            # Training each Pseudo GA classifier
            if len(self.ga_models) > 0:
                models = ModelsParams.build_models()
                for model in models:
                    model['model'].fit(X_train[:, self.ga_models[model['name'] + ' GA'].mask][:, ga_features[model['name']]], y_train)  # fit GA pseudo model
                    new_dataset_train.append(model['model'].predict(X_train[:, self.ga_models[model['name'] + ' GA'].mask][:, ga_features[model['name']]]))
                    new_dataset_test.append(model['model'].predict(X_test[:, self.ga_models[model['name'] + ' GA'].mask][:, ga_features[model['name']]]))
                    # print('Stacking: In GA Models')
            
            new_dataset = np.array(new_dataset_train, dtype=object)
            
            self.meta_model.fit(new_dataset.T, y_train)

            new_dataset_test = np.array(new_dataset_test, dtype=object)

            accuracy = accuracy_score(y_test, self.meta_model.predict(new_dataset_test.T))
            all_scores.append(accuracy)

        print(f'OSTACKING: {all_scores}')
        return np.mean(all_scores)



