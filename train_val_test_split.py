"""
File: train_val_test_split.py
Authors: Stan Martynov, Patrick Jans, Tieme Boerema

Description:
    This program implements a function that splits a given dataframe
    into a training, validation and test set without shuffling it.

"""

import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_test_split(data: pd.DataFrame) -> tuple[pd.Dataframe]:
    """
    Splits a given dataset into a training, validation, and a test set.
    Assumes the labels are the last column of the dataframe, and returns
    a non-shuffled 60-20-20 training-validation-test split.
    """
    # extract features and labels separately
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1:]

    # do two train_test_splits in order to create train/val/test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, shuffle=False
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, shuffle=False
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
