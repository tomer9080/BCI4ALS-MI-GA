"""
Parsers library - including function that parses for each part in pipeline
"""

import argparse


def parse_cmdl_offline():
    parser = argparse.ArgumentParser(description='This script is running classifiers on the requested folder, and then produces a table and csv file to compare between \n various classifiers and feature selection methods.')
    parser.add_argument('--folder', '-f', dest='folder', help='Folder path of the recording we want to classify', type=str)
    parser.add_argument('--folder2', '-f2', dest='folder2', help='Folder path of the second recording we want to classify - taken into account only if unify is True', type=str, default=None)
    parser.add_argument('--unify', '-u', dest='unify', help='Unify folder and folder2 to "one" recording and classify', type=bool)
    parser.add_argument('--paths', '-pa', dest='paths', help='Path to paths file to run classisfication for each folder in paths file', type=str, default=None)
    parser.add_argument('--metric', '-m', dest='metric', help='Metric string - according to him we\'ll choose our features by our statistic methods', type=str, default="Score_(R^2)_Left,Score_(R^2)_Right")
    parser.add_argument('--simple', '-s', dest='simple', help='Use simple metric (just sort dataframe by columns)', type=int, default=1)
    parser.add_argument('--prior', '-pr', dest='prior', help='How many prior recordings to look back. if no prior to be used - please enter 0. default value is 3', type=int, default=3)
    parser.add_argument('--new_folder', '-n', dest='new_folder', help='Where to save the stats in the class_results folder.', type=str)
    parser.add_argument('--correlation', '-c', dest='corr', help='Wether use correlation or not in choosing the features', type=bool, default=False)
    parser.add_argument('--ascending', '-a', dest='ascending', help='If using simple metric - sort in ascending order or not', type=bool, default=False)
    args = parser.parse_args()
    return {'folder': args.folder,
            'folder2': args.folder2,
            'unify': args.unify,
            'paths': args.paths,
            'metric': args.metric,
            'simple': args.simple,
            'prior': args.prior,
            'new_folder': args.new_folder,
            'corr': args.corr,
            'ascending': args.ascending}
