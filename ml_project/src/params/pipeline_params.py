from dataclasses import dataclass, field
from typing import List, Optional


@dataclass()
class SplitParams:
    val_size: float = field(default=0.2)
    random_state: int = field(default=1337)


@dataclass()
class FeatureParams:
    features_to_numerize: Optional[List[str]]
    features_to_drop: List[str]
    numerical_features: List[str]
    target_col: Optional[str]


@dataclass()
class TrainParams:
    model_type: str = field(default="RandomForestClassifier")
    n_estimators: int = field(default=100)
    random_state: int = field(default=1337)
