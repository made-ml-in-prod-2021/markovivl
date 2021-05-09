from argparse import ArgumentParser

import sys
import logging

from src.params import (
    PredictPipelineParams,
    read_predict_pipeline_params
)
from src.models import (
    model_load,
    model_predict
)
from src.data import (
    read_data,
    save_data,
    preds_to_df
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

DEFAULT_CONFIG = './configs/pred_config.yaml'


def setup_parser(parser, pred_params: PredictPipelineParams):
    """

    :param parser: argparse.ArgumentParser
    :param pred_params: PredictPipelineParams
    :return:
    """
    parser.add_argument(
        "--data",
        default=pred_params.input_data_path,
        dest="input_data_path",
        help="path to data",
    )

    parser.add_argument(
        "--model",
        default=pred_params.model_path,
        dest="model_path",
        help="path to model",
    )

    parser.add_argument(
        "--output",
        default=pred_params.output_path,
        dest="output_path",
        help="path to model",
    )


def setup_pipeline(config_path: str):
    params = read_predict_pipeline_params(config_path)
    return params


def predict(args):
    """
    Perform prediction
    :param args: input path, model path and output path
    """
    logger.info(f"Prediction start with following arguments {args}")
    data = read_data(args.input_data_path)
    logger.info(f"Successfully loaded dataset of shape {data.shape}")

    logger.info(f"Loading model from {args.model_path}")
    model = model_load(args.model_path)
    logger.info(f"Model loaded.")

    logger.info("Predicting...")
    preds = model_predict(model, data)
    save_data(preds_to_df(preds), args.output_path)
    logger.info(f"Predictions saved to {args.output_path}")


if __name__ == '__main__':
    parser = ArgumentParser()
    default_params = setup_pipeline(DEFAULT_CONFIG)
    setup_parser(parser, default_params)
    args = parser.parse_args()
    predict(args)
