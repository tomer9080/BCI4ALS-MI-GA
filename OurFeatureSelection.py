import numpy as np
import pandas as pd
import metrics_wrapper
import os

class Selector:

    def __init__(self, paths, record_path=None, features_names=metrics_wrapper.headers, ascending=True):
        """
        __init__ - initialize a selector instance
        paths - a path to the paths file that includes al of the recordings data.
        record_path - the record we want to extract features now - so we can know which prior recordings and knowledge to use - None will use last recording
        features_names - the names of all the features we have - usually an import from metrics wrapper.
        """
        self.paths = paths
        self.record_path = record_path
        self.names = features_names
        self.ascending = ascending
        self.types_dict = {key: float if key != 'Feature' else str for key in metrics_wrapper.table_headers}

    def get_index_by_name(self, name):
        return self.names.index(name)

    def get_recording_index_in_paths(self, paths_list: list):
        if self.record_path is None:
            return (len(paths_list) - 1)
        return paths_list.index(self.record_path.strip())

    def run_metrics(self, indices: tuple, paths_list: list, use_prior: bool):
        """
        run_metrics - run the metrics script according to the indices given one the relevant recordings
        indices - which recording should we take in account
        use_prior - use / not use prior.
        """
        if use_prior: # use prior recordings to analyze
            metrics_wrapper.analyze(paths_list[indices[0]:indices[1]], is_list=True)
        else:         # use only current recording
            metrics_wrapper.analyze([paths_list[indices[1] - 1]], is_list=True)

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
        paths_list = [line.strip() for line in paths_file.readlines()]
        current_index = self.get_recording_index_in_paths(paths_list)
        indices = (current_index - prior_recordings, current_index + 1)
        self.run_metrics(indices, paths_list, use_prior)
        # now we have the file stats/features_metrics.csv
        self.metrics = pd.read_csv('stats/features_metrics.csv')
        # this is simple metrics - ust ascending order by pandas sort.
        chosen_features = self.metrics.sort_values(metric_rule, ascending=self.ascending)['Feature'][:num_of_features]
        print(chosen_features)
        return [self.get_index_by_name(name) for name in chosen_features]

if __name__ == "__main__":
    selector = Selector('paths/paths_TK.txt')
    print(selector.select_features(['Var_Mean_left', 'Var_Mean_right']))