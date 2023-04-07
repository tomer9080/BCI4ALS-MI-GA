import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import OurUtils as Utils
from pathlib import Path

### ============= Models Scores ============ ###
def get_ga_scores():
    df = pd.read_csv('class_scores/full_ga_class_results/models_avg_scores.csv', delimiter=',')
    df.reset_index()
    scores_ga = {}
    scores_nca = {}
    for col in df.columns:
        if 'GA CV' in col:
            scores_ga[col.split()[0]] = df[col][0]
        elif 'NCA CV' in col:
            scores_nca[col.split()[0]] = df[col][0]
        elif 'NCA' in col:
            scores_nca[col.split()[0]] = df[col][0]
    return scores_nca, scores_ga

def get_sta_scores():
    df = pd.read_csv('stats/stats_scores.csv', delimiter=',')
    df.reset_index()
    scores_sta = {}
    models_names = ['LDA', 'QDA', 'SVC', 'RF', 'DT']
    models_names_no_cv = ['KNN-5', 'KNN-7', 'NB']
    
    for name in models_names:
        q_df = df[df['Model'].str.contains(name) & df['Model'].str.contains('CV')]
        s_df = q_df.sort_values(['Score'], ascending=False)
        print(s_df)
        scores_sta[name] = s_df.iloc[0]['Score']

    for name in models_names_no_cv:
        q_df = df[df['Model'].str.contains(name)]
        s_df = q_df.sort_values(['Score'], ascending=False)
        print(s_df)
        scores_sta[name] = s_df.iloc[0]['Score']

    print(scores_sta)
    return scores_sta

def get_models_scores():
    scores_nca, scores_ga = get_ga_scores()
    scores_sta = get_sta_scores()
    df = pd.DataFrame({
        'Classifier': list(scores_ga.keys()),
        'GA': list(scores_ga.values()),
        'NCA': list(scores_nca.values()),
        'STA': list(scores_sta.values())
    })

    df.plot(x='Classifier', y=['GA', 'NCA', 'STA'], kind='bar', title="Classifiers AVG scores comparison")
    plt.ylabel('Score(%)')
    plt.ylim((0.4,0.7))
    plt.tight_layout()
    plt.savefig('stats\\class_comparisons', dpi=600)
    plt.show()


### ============= GA FEATURES ============= ###
def get_ga_features() -> dict():
    all_hists = {}
    for subdir, _, _ in os.walk('ga_features'):
        if subdir == 'ga_features':
            continue
        print(subdir)
        if subdir.count('\\') < 2:
            continue
        for _, _, files_s in os.walk(subdir):
            hist = {}
            subdir_list = Path(subdir).parts
            model_name = subdir_list[-1]
            for file in files_s:
                features_file = open(f'{subdir}\\{file}', 'rt')
                lines = [line.strip() for line in features_file]
                for line in lines:
                    if hist.get(line):
                        hist[line] += 1
                    else:
                        hist[line] = 1
            hist = {key: item for key, item in hist.items()}
            all_hists[model_name] = hist
    Utils.save_dict_to_pickle(all_hists, 'ga_models_features_hists')
    return all_hists

def show_hist_ga():
    all_hist = get_ga_features()

    for model, hist in all_hist.items():
        plt.bar(hist.keys(), hist.values(), color='b')
        plt.title(f'Most selected features by GA on {model} model')
        plt.xlabel('Feature')
        plt.ylabel('Count')
        plt.ylim((2,6.5))
        plt.xticks(fontsize=5, rotation=80)
        plt.yticks(np.arange(2, 6.5, step=1))
        plt.tight_layout()
        plt.savefig(f'ga_hists/{model}_hist', dpi=600)


if __name__ == "__main__":
    # show_hist_ga()
    # get_models_scores()
    print(get_ga_features())