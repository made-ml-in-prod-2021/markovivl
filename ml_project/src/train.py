from argparse import ArgumentParser

import sys
import logging
import json

from src.params import (
    TrainPipelineParams,
    read_train_pipeline_params
)
from src.features.build_features import (
    extract_target,
    build_transformer,
)
from src.models import (
    model_build,
    model_train,
    model_predict,
    model_evaluate,
    model_save,
)
from src.data import(
    read_data,
    split_train_val_data
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

DEFAULT_CONFIG = '../configs/train_gbc_config.yaml'


def setup_parser(parser):
    """
    Setup config on a given parser
    :param parser: argparse.ArgumentParser
    """
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        dest="config_path",
        help="path to yaml config",
    )


def setup_pipeline(config_path: str):
    """
    Setup pipeline from a given config file
    :param config_path: Path to a .yaml config
    :return: pipeline params
    """
    params = read_train_pipeline_params(config_path)
    return params


def train_pipeline(train_pipeline_params: TrainPipelineParams):
    """
    Read and split data, train and validate model, save model and metrics
    :param train_pipeline_params:
    :return: path to serialized model and metrics
    """
    logger.info(f"starting training with following parameters {train_pipeline_params}")
    data = read_data(train_pipeline_params.input_data_path)
    logger.info(f"data loaded from {train_pipeline_params.input_data_path}")
    train_data, val_data = split_train_val_data(data, train_pipeline_params.split_params)

    transformer = build_transformer(train_pipeline_params.feature_params)
    train_target = extract_target(train_data, train_pipeline_params.feature_params)
    logger.info(f"Extracted target with the following shape {train_target.shape}")

    logger.info("Beginning training...")
    clf_model = model_build(transformer, train_params=train_pipeline_params.train_params)
    model = model_train(clf_model, train_data, train_target)
    logger.info("Finishing training...")

    logger.info("Beginning validation...")
    val_target = extract_target(val_data, train_pipeline_params.feature_params)

    preds = model_predict(model, val_data)
    logger.info("Scoring the model on the evaluation...")
    metrics = model_evaluate(val_target, preds)

    logger.info(f"Metrics \n {metrics}")
    with open(train_pipeline_params.metric_path, "w") as metrics_file:
        json.dump(metrics, metrics_file)
    logger.info(f"Metrics are saved to {train_pipeline_params.metric_path}")

    path_to_model = model_save(model, train_pipeline_params.output_model_path)
    logger.info(f"Model was saved to {train_pipeline_params.output_model_path}")
    return path_to_model, metrics


if __name__ == '__main__':
    parser = ArgumentParser()
    setup_parser(parser)
    arguments = parser.parse_args()
    params = setup_pipeline(arguments.config_path)
    train_pipeline(params)
