"""
Parsers library - including function that parses for each part in pipeline
"""
import argparse
import OurUtils as Utils

class globs:
    
    args_dict = {}

    def __init__(self) -> None:
        pass

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
    parser.add_argument('--grid', '-gr', dest='grid', help='run grid search on classifiers', type=bool, default=False)
    parser.add_argument('--genetic_algorithm', '-ga', dest='genetic', help='run ga classification', type=bool, default=False)
    parser.add_argument('--save_models', '-sm', dest='save_models', help='save models to pickle files', type=bool, default=False)
    parser.add_argument('--expanded', '-ex', dest='expanded', help='Use expanded features to run with', type=bool, default=False)
    parser.add_argument('--threshold', '-th', dest='threshold', help='Must run with GA. Reduces search spaceto search for features that have been chosen before >= the threshold', type=int, default=0)
    args = parser.parse_args()
    globs.args_dict = {'folder': args.folder,
                'folder2': args.folder2,
                'unify': args.unify,
                'paths': args.paths,
                'metric': args.metric,
                'simple': args.simple,
                'prior': args.prior,
                'new_folder': args.new_folder,
                'corr': args.corr,
                'ascending': args.ascending,
                'grid': args.grid,
                'ga': args.genetic,
                'save_models': args.save_models,
                'expanded': args.expanded,
                'index_max': Utils.get_index_max(),
                'threshold': args.threshold
                }
    return globs.args_dict

def parse_cmdl_online():
    parser = argparse.ArgumentParser(description='This script is running classifiers on the requested folder, and then produces a table and csv file to compare between \n various classifiers and feature selection methods.')
    parser.add_argument('--offline', '-off', dest='offline', help='Folder path of the models that were trained offline', type=str)
    parser.add_argument('--online', '-on', dest='online', help='Folder path of the recording we want to run online classification on', type=str, default=None)
    parser.add_argument('--name', '-n', dest='name', help='Folder name of which run we want to take chosen features from', type=str, default=None)
    parser.add_argument('--threshold', '-th', dest='threshold', help='Must run with GA. Reduces search space to search for features that have been chosen before >= the threshold', type=int, default=0)
    args = parser.parse_args()
    globs.args_dict = {'offline': args.offline,
                'online': args.online,
                'name': args.name,
                'threshold': args.threshold
                }
    return globs.args_dict

def parse_cmdl_studies():
    parser = argparse.ArgumentParser(description='This script is running analysis using contour plots.')
    parser.add_argument('--model', '-m', dest='model', help='The Model we want the data from', type=str)
    parser.add_argument('--attr1', '-a1', dest='attr1', help='Folder path of the recording we want to run online classification on', type=str, default=None)
    parser.add_argument('--attr2', '-a2', dest='attr2', help='Folder name of which run we want to take chosen features from', type=str, default=None)
    args = parser.parse_args()
    globs.args_dict =  {'model': args.model,
                        'attr1': args.attr1,
                        'attr2': args.attr2
                        }
    return globs.args_dict

def get_args_dict():
    return globs.args_dict
