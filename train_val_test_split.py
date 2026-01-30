"""
File: train_val_test_split.py
Authors: Stan Martynov, Patrick Jans, Tieme Boerema

Description:
    This program implements a function that splits a given dataframe
    into a training, validation and test set without shuffling it.

"""


def train_val_test_split(data):
    """
    Splits a given dataset into a training, validation, and a test set.
    Assumes the labels are the last column of the dataframe, and returns
    a non-shuffled 60-20-20 training-validation-test split.
    """
    # extract features and labels separately
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1:]

    # We split by these dates
    training_end = 20060101.0
    val_end = 20080101.0

    # We cannot use train_test split since then we have data from the same
    # day overlapping in the training, test and val set which can cause leakage
    train_index = X["DATE"] < training_end
    val_index = (X["DATE"] >= training_end) & (X["DATE"] < val_end)
    test_index = X["DATE"] >= val_end

    X_train, y_train = X[train_index], y[train_index]
    X_val, y_val = X[val_index], y[val_index]
    X_test, y_test = X[test_index], y[test_index]

    X_train = X_train.drop(columns=["DATE"])
    X_val = X_val.drop(columns=["DATE"])
    X_test = X_test.drop(columns=["DATE"])

    return X_train, X_val, X_test, y_train, y_val, y_test
