from typing import Tuple, List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from ..params import SplitParams


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def split_train_val_data(
        data: pd.DataFrame,
        params: SplitParams,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    return train_df, val_df


def save_data(
        data: pd.DataFrame,
        output_path: str
) -> None:
    data.to_csv(output_path, index=False)


def preds_to_df(
        preds: List or np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame({"preds": preds})
