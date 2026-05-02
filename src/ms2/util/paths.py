### ~~~ GLOBAL IMPORTS ~~~ ###
from pathlib import Path
from enum import Enum

### ~~~ LOCAL IMPORTS ~~~ ###
# None


### ~~~ STATE MANAGEMENT ~~~ ###
class DataSteps(Enum):
    external = "external"
    interim = "interim"
    processed = "processed"
    artifacts = "artifacts"


class PipelineStep(Enum):
    data_preprocessing = "data_preprocessing"
    model_training = "model_training"
    model_evaluation = "model_evaluation"


data_path: dict[DataSteps, Path] = {
    DataSteps.external: Path("data/external/ms2"),
    DataSteps.interim: Path("data/interim/ms2"),
    DataSteps.processed: Path("data/processed/ms2"),
    DataSteps.artifacts: Path("data/artifacts/ms2"),
}

config_path: dict[PipelineStep, Path] = {
    PipelineStep.data_preprocessing: Path("configs/ms2/preprocessing.yml")
}


__all__ = ["DataSteps", "data_path", "PipelineStep", "config_path"]
