import numpy as np
import pandas as pd


def mse(ratings: pd.Series, prediction: pd.Series) -> float:
    loss = ((ratings - prediction) ** 2).mean()
    return np.sqrt(loss)
