from .pipeline_params import SplitParams, FeatureParams, TrainParams
from dataclasses import dataclass
from marshmallow_dataclass import class_schema

import yaml


@dataclass()
class TrainPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    split_params: SplitParams
    feature_params: FeatureParams
    train_params: TrainParams


@dataclass()
class PredictPipelineParams:
    input_data_path: str
    model_path: str
    output_path: str


TrainPipelineParamsSchema = class_schema(TrainPipelineParams)

PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_train_pipeline_params(path: str) -> TrainPipelineParams:
    with open(path, "r") as input_config:
        schema = TrainPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_config))


def read_predict_pipeline_params(path: str):
    with open(path, "r") as input_config:
        schema = PredictPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_config))
