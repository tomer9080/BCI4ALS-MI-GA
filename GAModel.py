import os
import OurUtils as Utils
from genetic_selection import GeneticSelectionCV
from Parsers import get_args_dict

headers = Utils.headers

class GAModel:

    def __init__(self, name, estimator, cv=3, scoring="accuracy", max_features=10, n_population=153, cross_prob=0.5, muta_prob=0.2, n_gens=60, caching=True, muta_ind_prob=0.025, cross_ind_prob=0.8, threshold=0):
        self.name = name
        self.selector = GeneticSelectionCV(estimator, cv, scoring, max_features=max_features, n_population=n_population, crossover_proba=cross_prob, mutation_proba=muta_prob, n_generations=n_gens, caching=caching, mutation_independent_proba=muta_ind_prob, crossover_independent_proba=cross_ind_prob)
        self.threshold = threshold

    def fit(self, X, y):
        X_red, mask = self.reduce_ga_search_space(X, self.name)
        self.mask = mask
        self.selector.fit(X_red, y)
        self.support_ = self.selector.support_
        return self

    def predict(self, X):
        X_red, _ = self.reduce_ga_search_space(X, self.name)
        return self.selector.predict(X_red)

    def reduce_ga_search_space(self, features, model_name):
        threshold = get_args_dict().get('threshold', 0) if self.threshold == 0 else self.threshold
        hists_ga: dict = Utils.load_from_pickle(os.path.join('stats', 'ga_models_features_hists'))
        hist: dict = hists_ga.get(model_name)
        if hist is None:
            return features, [True] * len(headers)
        mask = [(feature in hist.keys() and hist.get(feature, 0) >= threshold) for feature in headers]
        # print(f"Threshold is {threshold}, model: {model_name}, features[:, mask] size: {(features[:,mask]).shape}")
        return features[:,mask], mask

    def predict_proba(self, X):
        X_red, _ = self.reduce_ga_search_space(X, self.name)
        return self.selector.predict_proba(X_red)
    
    def score(self, X, y):
        X_red, _ = self.reduce_ga_search_space(X, self.name)
        return self.selector.score(X_red, y)
    
    def get_scores(self):
        return self.selector.generation_scores_

    
