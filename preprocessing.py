"""
File: preprocessing.py
Authors: Stan Martynov, Patrick Jans, Tieme Boerema

Description:
    This program reformulates the dataset "weather_prediction.csv" into
    a processed version which does not group the features by the cities
    in which they were observed.

Comments:
    There are surely better/faster ways to get this done using pandas,
    e.g. using df.melt, but we were unable to get this to work and chose to
    create this instead. Given the fact that this is preprocessing and that
    the program only has to be run once, the suboptimality is not particularly
    damaging.

"""

import numpy as np
import pandas as pd

# define global variables cities and measurements
cities = [
    "BASEL",
    "BUDAPEST",
    "DE_BILT",
    "DRESDEN",
    "DUSSELDORF",
    "HEATHROW",
    "KASSEL",
    "LJUBLJANA",
    "MAASTRICHT",
    "MALMO",
    "MONTELIMAR",
    "MUENCHEN",
    "OSLO",
    "PERPIGNAN",
    "ROMA",
    "SONNBLICK",
    "STOCKHOLM",
    "TOURS",
]

measurements = [
    "cloud_cover",
    "wind_speed",
    "wind_gust",
    "humidity",
    "pressure",
    "global_radiation",
    "precipitation",
    "sunshine",
    "temp_mean",
    "temp_max",
    "temp_min",
]


def convert_row(row: pd.Series, city: str) -> list[np.float64]:
    """
    Given a dataframe row and a city to consider, finds all measurements
    corresponding to this city in the given row, and creates a new row
    based on these measurements.
    """
    new_row = [row["DATE"], row["MONTH"]]
    for measurement in measurements:
        col_name = city + "_" + measurement
        if col_name not in row:
            new_row.append(np.nan)
        else:
            new_row.append(row[col_name])

    return new_row


def create_new_df(data: list[list]) -> pd.DataFrame:
    """
    Creates a new dataframe based on list data
    """
    columns = ["DATE", "MONTH"] + measurements
    df = pd.DataFrame(data, columns=columns)

    return df


def main() -> None:
    df_data = pd.read_csv("datasets/weather_prediction_dataset.csv")

    # initialize new dataframe
    processed_df = pd.DataFrame()

    new_rows = []
    for idx, row in df_data.iterrows():
        for city in cities:
            new_rows.append(convert_row(row, city))

    processed_df = create_new_df(new_rows)
    processed_df.to_csv("datasets/processed_dataset.csv", index=False)


if __name__ == "__main__":
    main()
