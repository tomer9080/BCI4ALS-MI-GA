import OurUtils as Utils
import numpy as np
import ModelsUtils
import ModelsParams
from sklearn.model_selection import cross_validate
from tabulate import tabulate
from Parsers import parse_cmdl_online

def run_offline_models(args_dict: dict):
    features, labels, _, nca_indices = Utils.get_all_features(args_dict['offline'])
    models = ModelsParams.build_models()
    rows = []
    for model in models:

        rows.append([model['name'] + ' CV',  np.mean(cross_validate(model['model'], features[:, nca_indices], labels, scoring='accuracy')['test_score'])])

        _, mask = ModelsUtils.reduce_ga_search_space(features, model.get('name'), model.get('thresh'))
        rows.append([model['name'] + ' GA CV', np.mean(cross_validate(model['model'], features[:, mask], labels, scoring='accuracy')['test_score'])])

    print('')
    table_headers = ["Classifier", "Success Rate"]
    print(tabulate(rows, headers=table_headers, tablefmt='grid'))

if __name__ == "__main__":
    args_dict = parse_cmdl_online()
    run_offline_models(args_dict)