import pandas as pd
import numpy as np


def data_preprocessing(df_dataset: pd.DataFrame, normalized: bool, one_hot: bool):
    """
    function that generate X, t
    """
    if normalized:
        X = ((df_dataset.iloc[:,2:] - df_dataset.iloc[:,2:].min()) / (df_dataset.iloc[:,2:].max() - df_dataset.iloc[:,2:].min())).values
    else:
        X = df_dataset.iloc[:, 2:].values

    species = df_dataset['species'].unique()

    if one_hot:
        # Creation of one-hot vectors for the target matrix
        t = np.zeros((len(X), species.size))
        for i in range(species.size):
            t[df_dataset['species'] == species[i], i] = 1
    else:
        # Generate a target vector. Each vector correspond to number associate to the class.
        t = np.zeros(df_dataset.shape[0])
        for i in range(species.size):
            t[df_dataset['species'] == species[i]] = i
    return X, t
