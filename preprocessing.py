"""
File: preprocessing.py
Authors: Stan Martynov, Patrick Jans, Tieme Boerema

Description:
    This program reformulates the dataset "weather_prediction.csv" into
    a processed version which does not contain the names of the cities,
    in which the data was gathered.

"""

import re

import pandas as pd


def main():
    df_data = pd.read_csv("datasets/weather_prediction_dataset.csv")
    print(df_data.describe)

    processed_df = pd.Dataframe()
    for datapoint in df_data:
        pass

    # processed_df.to_csv("processed_dataset")


if __name__ == "__main__":
    main()
