import numpy as np
import pandas as pd
import metrics_wrapper

class Selector:

    def __init__(self, paths, features_names):
        self.paths = paths
        self.names = features_names
        self.types_dict = {key: float if key != 'Feature' else str for key in metrics_wrapper.table_headers}

    def get_index_by_name(self, name):
        return self.names.index(name)

    def select_features(self, metric_rule, threshold=1, num_of_features=10, use_prior=True, prior_recordings=3):
        """
        select_features - a function that given a metric, and a threshold tries to select to most 
        stationary features according to this metric, and return them according to a prior data if requested.
        if not - the function analyzes the features stationarity in the recording space only, not looking at
        any prior.
        params:
        metric_rule - a tuple of criterions, by them we calculate the stationarity of a feature.
        right now will be only a list that we'll ascned the dataframe by it.
        threshold - the threshold for a feature to be marked as stationary according to the metric.
        num_of_features - number of features you'll get from the function. if -1 is given, all of the values above the thershold will be returned.
        use_prior - bool value that determines if selector using prior data.
        prior_recordings - integer, determines how back (in recordings sacle) we want to look using our prior.
        """
        # first we need to run metrics_wrapper so we'll get some metrics.
        paths_file = open(self.paths, 'r')
        paths_list = paths_file.readlines()
        if use_prior: # use prior recordings to analyze
            metrics_wrapper.analyze(paths_list[-prior_recordings:], is_list=True)
        else:         # use only current recording 
            metrics_wrapper.analyze(paths_list[-1], is_list=True)
        # now we have the file stats/features_metrics.csv
        self.metrics = pd.read_csv('stats/features_metrics.csv')
        # this is simple metrics - ust ascending order by pandas sort.
        chosen_features = self.metrics.sort_values(metric_rule)['Feature'][:num_of_features]
        print(chosen_features)
        return [self.get_index_by_name(name) for name in chosen_features]

if __name__ == "__main__":
    selector = Selector('paths/paths_TK.txt', metrics_wrapper.headers)
    print(selector.select_features(['Var_Mean_left', 'Var_Mean_right']))