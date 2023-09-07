import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import OurUtils as Utils
from pathlib import Path
from OurUtils import from_feature_name_to_index

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
def get_ga_features(start=0) -> dict():
    all_hists = {}
    for subdir, _, _ in os.walk('ga_features'):
        print(subdir)
        if subdir == 'ga_features':
            continue
        if subdir.count('\\') + subdir.count('/') < 2:
            continue
        for _, _, files_s in os.walk(subdir):
            hist = {}
            print(subdir)
            subdir_list = Path(subdir).parts
            model_name = subdir_list[-1]
            for file in files_s:
                features_file = open(f'{os.path.join(subdir, file)}', 'rt')
                lines = [line.strip() for line in features_file]
                for line in lines:
                    if hist.get(line):
                        hist[line] += 1
                    else:
                        hist[line] = 1
            hist = {key: item for key, item in hist.items()}
            if all_hists.get(model_name):
                all_keys = set(list(hist.keys()) + list(all_hists[model_name].keys()))
                all_hists[model_name] = {key: all_hists[model_name].get(key, 0) + hist.get(key, 0) for key in all_keys}
            else:
                all_hists[model_name] = hist
    Utils.save_dict_to_pickle(all_hists, 'ga_models_features_hists')
    return all_hists

def show_hist_ga():
    all_hist = get_ga_features()

    for model, hist in all_hist.items():
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        filtered_hist = {key: item for key, item in hist.items() if item >= 100}
        sorted_filtered = {key: val for key, val in sorted(filtered_hist.items(), key=lambda ele: ele[1])}
        ax.bar(sorted_filtered.keys(), list(sorted_filtered.values()), color='b')
        print(f'{model}: {list(hist.values())}')
        ax.set_title(f'Most selected features by GA on {model} model')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Count')
        fig.tight_layout()
        ax.set_xticks(ticks=list(sorted_filtered.keys()))
        ax.set_xticklabels(list(sorted_filtered.keys()), rotation=45, fontsize=5)
        fig.savefig(f'ga_hists/{model}_hist', dpi=600)
        fig.clear()

        #save top ten features to a different figure
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        top_ten = list(sorted_filtered.keys())[-10:]
        top_ten_vals = [sorted_filtered[key] for key in top_ten]
        ax.bar(top_ten, top_ten_vals, color='b')
        ax.set_title(f'Top ten features selected by GA on {model} model')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Count')
        fig.tight_layout()
        ax.set_xticks(ticks=top_ten)
        ax.set_xticklabels(top_ten, rotation=45, fontsize=5)
        fig.savefig(f'top_ten_features/{model}_top_ten_features', dpi=600)
        fig.clear()
        
    
if __name__ == "__main__":
    # show_hist_ga()
    # get_models_scores()
    show_hist_ga()