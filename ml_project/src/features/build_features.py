import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

from src.params import FeatureParams


def extract_target(
        data: pd.DataFrame,
        params: FeatureParams
) -> pd.Series:
    return data[params.target_col]


def build_ftn_pipeline() -> Pipeline:
    ftn_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("label_encode", OrdinalEncoder()),
        ]
    )
    return ftn_pipeline


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("impute", KNNImputer(n_neighbors=9))
        ]
    )
    return num_pipeline


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "ftn_pipeline",
                build_ftn_pipeline(),
                params.features_to_numerize,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            ),
            (
                "drop_pipeline",
                "drop",
                params.features_to_drop,
            ),
        ]
    )
    return transformer
