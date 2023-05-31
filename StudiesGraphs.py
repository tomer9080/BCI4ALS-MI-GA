import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path
from Parsers import parse_cmdl_studies


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

    def contour_plot(self, attr1, attr2):
        relevant = self.all_studies[['value', attr1, attr2]].to_numpy()
        X, Y = np.meshgrid(relevant[:, 1], relevant[:, 2])
        print(X.shape, Y.shape)
        fig = px.density_contour(self.all_studies, x=attr1, y=attr2, z='value')
        fig.update_traces(contours_coloring="fill", contours_showlabels=True)
        fig.show()

        plt.tricontourf(relevant[:,1], relevant[:,2], relevant[:,0])
        plt.scatter(relevant[:,1], relevant[:,2], relevant[:,0], c='r')
        plt.colorbar()
        plt.xlabel(attr1)
        plt.ylabel(attr2)
        plt.title(f"Accuracy vs {attr1}, {attr2}")
        plt.show()


        


def run():
    # args_dict = parse_cmdl_studies()
    sp = StudyPlotter('LDA')
    sp.contour_plot('params_muta_prob', 'params_cross_prob')

if __name__ == "__main__":
    run()