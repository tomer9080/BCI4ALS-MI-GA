import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path
from Parsers import parse_cmdl_studies
from ModelsParams import build_models


class StudyPlotter:

    def __init__(self, model) -> None:
        self.model = model
        self.pre_process_studies()

    def pre_process_studies(self):
        if Path(os.path.join('studies', self.model, 'all_studies.csv')).exists():
            self.all_studies = pd.read_csv(os.path.join('studies', self.model, 'all_studies.csv'))
            return
        
        root_dir = os.path.join('studies', self.model)
        dfs = []
        # Traverse through the directory tree
        for root, directories, files in os.walk(root_dir):
            for file in files:
                # Check if the file is a CSV
                if file.endswith('.csv'):
                    # Construct the absolute path to the CSV file
                    csv_path = os.path.join(root, file)
                    
                    # Read the CSV file into a DataFrame
                    df = pd.read_csv(csv_path)
                    
                    # Append the DataFrame to the list
                    dfs.append(df)

        # Concatenate all DataFrames into a single DataFrame
        unified_df = pd.concat(dfs)

        unified_df.to_csv(os.path.join(root_dir, 'all_studies.csv'))

        self.all_studies = unified_df

    def plot_hist_contour(self, attr1, attr2):        
        fig = px.density_contour(self.all_studies, x=attr1, y=attr2, z='value')
        fig.update_traces(contours_coloring="fill", contours_showlabels=True)
        fig.show()

    def contour_plot(self, attr1, attr2):
        relevant = self.all_studies[['value', attr1, attr2]].sort_values('value', ascending=False).head(270).to_numpy()
        print(relevant)
        ps = plt.tricontourf(relevant[:,1], relevant[:,2], relevant[:,0], vmin=np.min(relevant[:,0]-0.3), vmax=np.max(relevant[:,0]))
        plt.scatter(relevant[:,1], relevant[:,2], relevant[:,0], c='r')
        plt.colorbar(ps)
        plt.xlabel(attr1)
        plt.ylabel(attr2)
        plt.title(f"Accuracy vs {attr1}, {attr2}")
        plt.savefig(os.path.join('studies', self.model, 'contours', f'{self.model}_{attr1}_{attr2}.png'))
        plt.clf()

    def contour_all_possible_plots(self):
        cols = self.all_studies.columns.to_numpy()
        cols = list(cols[['params' in col for col in cols]])
        pairs = []
        for i in range(len(cols)):
            for j in range(len(cols) - i):
                if j+i == i:
                    continue
                pairs.append((cols[i], cols[j+i]))
        for pair in pairs:
            self.contour_plot(pair[0], pair[1])


def run():
    models = build_models()
    for model in models[:1]:
        sp = StudyPlotter(model['name'])
        sp.contour_all_possible_plots()


if __name__ == "__main__":
    run()