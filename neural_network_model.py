"""
File: neural_network_model.py
Authors: Stan Martynov, Patrick Jans, Tieme Boerema

Description:
    This program trains and tunes an MLP/FFNN to perform binary
    classification on the weather prediction dataset.

"""

import keras
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, fbeta_score
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense

from train_val_test_split import train_val_test_split


def evaluate_model(model: Sequential, X_val: pd.DataFrame, y_val: pd.DataFrame) -> None:
    """
    TODO
    """
    y_val_pred = model.predict(X_val)
    y_val_pred = y_val_pred > 0.5

    print(f"Accuracy on validation set: {accuracy_score(y_val, y_val_pred)}")
    print(f"F2-score on validation set: {fbeta_score(y_val, y_val_pred, beta=2)}")


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
) -> Sequential:
    """
    Trains a FFNN on given training data, and uses Early Stopping to
    combat potential overfitting.
    """
    model = Sequential()
    model.add(Input(shape=(10,)))
    model.add(Dense(15, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    train_history = model.fit(
        X_train,
        y_train,
        epochs=200,
        validation_data=(X_val, y_val),
        callbacks=earlystop,
    )

    return model


def main() -> None:
    """
    Splits data into training, test, validation splits, then uses it to train
    a Feed-Forward Neural Network (FFNN), which is evaluated on a validation set.
    """
    # set random seed to ensure reproducibility
    keras.utils.set_random_seed(42)

    data = pd.read_csv("datasets/processed_dataset.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(data)

    X_train, X_val, X_test = X_train.iloc[:, 2:], X_val.iloc[:, 2:], X_test.iloc[:, 2:]
    model = train_model(X_train, y_train, X_val, y_val)
    evaluate_model(model, X_val, y_val)


if __name__ == "__main__":
    main()
