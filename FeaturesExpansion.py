"""
Use inner products to increase the number of features to be used and to choose from.
"""
import numpy as np
from OurUtils import get_all_features
from pathlib import Path


def inner_product_features(X: np.ndarray, recording: str):
    if Path(f"{recording}\\expanded_features.csv").is_file():
        return np.loadtxt(f"{recording}\\expanded_features.csv", delimiter=',')
    ijs = []
    expanded = []
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            if (i,j) in ijs or (j, i) in ijs:
                continue
            cols_product = X[:,i] * X[:,j]
            expanded.append(cols_product)
            ijs.append((i, j))
    stacked = np.concatenate((X, np.array(expanded).T), axis=1)
    print(stacked.shape)
    np.savetxt(f"{recording}\\expanded_features.csv", stacked, delimiter=',')
    

def radial_basis_features(X: np.ndarray, recording):
    print(X.shape[1])
    if Path(f"{recording}\\radial_features.csv").is_file():
        return np.loadtxt(f"{recording}\\radial_features.csv", delimiter=',')
    expanded = []
    for i in range(X.shape[1]):
        c = 0.5
        radial_value = np.exp(-((X[:,i] - c)**2))
        expanded.append(radial_value)
    stacked = np.concatenate((X, np.array(expanded).T), axis=1)
    print(stacked.shape)
    np.savetxt(f"{recording}\\radial_features.csv", stacked, delimiter=',')
    

def expand_features(recording):
    """
    Here we put all the features manipulations functions.
    """
    all_features: np.ndarray = get_all_features(recording)[0]
    inner_features = inner_product_features(X=all_features, recording=recording)
    radial_features = radial_basis_features(X=all_features, recording=recording)
    return radial_features
    

if __name__ == "__main__":
    print(expand_features(r"Recordings\16-08-22\TK\Sub318324886001"))