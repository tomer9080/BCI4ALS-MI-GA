import os
import sys
import pandas as pd
import numpy as np

def get_sta_scores(path):
    """
    get_sta_scores - returns all of the scores of statistical features based models.
    path - the class_scores run path
    """
    stats_scores: pd.DataFrame = pd.read_csv(f"{path}\\models_avg_scores.csv")
    print(stats_scores)
    return stats_scores

def analyze_scores():
    """
    analyze_scores - prints to a file the scores of each statistical model sorted from best to worst.
    """
    root_path = 'class_scores'
    all_scores = {}
    for (root, dirs, files) in os.walk(root_path):
        for dir in dirs:
            if 'unified' in dir:
                continue
            scores = get_sta_scores(f'{root_path}\\{dir}')
            for header in scores.columns:
                if 'STA' in header:
                    all_scores[f'{dir}_{header}'] = scores[header][0]
    sta_scores = open("./stats/stats_scores.txt", 'wt')
    all_scores_sorted = dict(sorted(all_scores.items(), key=lambda x:x[1]))
    print(all_scores_sorted)
    for key, val in all_scores_sorted.items():
        sta_scores.write(f"{key}: {val}\n")
    sta_scores.close()


if __name__ == "__main__":
    analyze_scores()