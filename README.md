<h1>Intro to ML - Final Project</h1>

In this assignment, we aim to create a machine learning model that is able to correctly predict whether there will be precipitation on a given day, based on other measurements, such as cloud cover, wind speed, minimum temperature, and level of sunshine. To do this, we make use of the [weather prediction dataset](https://github.com/florian-huber/weather_prediction_dataset/tree/main). In this README, we will outline how our project results can be reproduced.

<h3> 1 - First, navigate to an empty directory and clone the repository </h3>

```
git clone https://github.com/tiemeboerema/intro_ML_Final_Project
```

<h3> 2a - If using a uv environment </h3>

```
uv sync
```

<h3> 2b - otherwise, create a virtual environment and install the dependencies </h3>

If possible, use Python 3.11, as this is the version the project was created and tested on. We cannot guarantee reproducibility if you are using a different version.

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

<h3> 3 - Run the preprocessing script to transform the dataset </h3>

```
python3 preprocessing.py
```

<h3> 4 - "Run All" on the four model notebooks </h3>

One notebook is included for each of the three ML models we trained, plus one more notebook for the final evaluation of the best model on our test set.