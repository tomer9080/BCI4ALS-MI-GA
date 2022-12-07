"""
Online -  this module is a script that can run an asked model, given its weights, and the features selected,
and run and classify the wanted recodings given by a file / list.
TODO: Make adjustments using quarter of the samples (fine tuning for online learning)
TODO: be able to use any classifier.
TODO: use pickle to load the classifier of the desired run
TODO: use parser to tell 
"""

import pickle

if __name__ == "__main__":
    model = pickle.load(open('tmp\\LDA_object.pkl', 'rb'))
    print(model.params)