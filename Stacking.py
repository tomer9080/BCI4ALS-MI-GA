import numpy as np
import ModelsUtils

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
        self.nca_features_train = (features[self.train_indices,:])[:,nca_indices]
        self.nca_features_test = (features[test_indices,:])[:,nca_indices]
        self.meta_model = level1_model
    
    def get_ga_features(self, model, name: str, train=True):
        reduced_features, mask = ModelsUtils.reduce_ga_search_space(features=self.features, model_name=name.replace(' GA', ''))
        to_ret = (reduced_features[self.train_indices,:]) if train else (reduced_features[self.test_indices,:])
        print(name)
        print(to_ret.shape)
        return to_ret

    def fit(self):

        # Training the meta_model
        new_dataset_train = []
        new_dataset_test = []
        
        for _, model in self.reg_models.items():
            new_dataset_train.append(model.predict(self.nca_features_train))
            new_dataset_test.append(model.predict(self.nca_features_test))

        for key, model in self.ga_models.items():
            new_dataset_train.append(model.predict(self.get_ga_features(model, key)))
            new_dataset_test.append(model.predict(self.get_ga_features(model, name=key, train=False)))

        new_dataset = np.array(new_dataset_train, dtype=object)
        self.new_dataset = new_dataset.T  # Transposed to make rows to columns.
        print(self.new_dataset.shape)  # should be from shape: (num_train_trials, num_models)

        self.meta_model.fit(self.new_dataset, self.train_labels)

        self.new_dataset_test = np.array(new_dataset_test, dtype=object).T

    def predict(self):
        return self.meta_model.predict(self.new_dataset_test)