from typing import Dict
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.params import TrainParams


def model_build(
        transformer: TransformerMixin,
        train_params: TrainParams,
) -> Pipeline:
    if train_params.model_type == "GradientBoostingClassifier":
        model = GradientBoostingClassifier(random_state=train_params.random_state)
    elif train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=train_params.n_estimators,
            random_state=train_params.random_state,
        )
    else:
        raise NotImplementedError()

    fin_model = Pipeline(
        [
            ("transformer", transformer),
            ("classifier", model),
        ]
    )
    return fin_model


def model_train(
        model: Pipeline,
        data: pd.DataFrame,
        target: pd.Series,
) -> Pipeline:
    model.fit(data, target)
    return model


def model_predict(
        model: Pipeline,
        data: pd.DataFrame,
) -> np.ndarray:
    preds = model.predict(data)
    return preds


def model_evaluate(
        target: pd.Series,
        preds: np.ndarray
) -> Dict[str, float]:
    evaluation = {
        "accuracy": accuracy_score(target, preds),
        "precision": precision_score(target, preds),
        "recall": recall_score(target, preds),
        "f1": f1_score(target, preds),
    }
    return evaluation


def model_save(
        model: Pipeline,
        filename: str,
) -> str:
    with open(filename, "wb") as model_dump:
        pickle.dump(model, model_dump)
    return filename


def model_load(
        filename: str
) -> Pipeline:
    with open(filename, "rb") as load_model:
        model = pickle.load(load_model)
    return model
