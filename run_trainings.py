import sys
sys.path.append('C:\BCIToolBox\MatlabCode')
import MI5_ModelTraining
from metrics_wrapper import get_paths

if __name__ == '__main__':
    paths = get_paths()
    for path in paths:
        print(f'running now classify on path: {path}')
        MI5_ModelTraining.classify(recordingFolder=path, recordingFolder_2='')
