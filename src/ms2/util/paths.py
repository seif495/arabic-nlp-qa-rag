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


data_path: dict[DataSteps, Path] = {
    DataSteps.external: Path("data/external/ms2"),
    DataSteps.interim: Path("data/interim/ms2"),
    DataSteps.processed: Path("data/processed/ms2"),
    DataSteps.artifacts: Path("data/artifacts/ms2"),
}


__all__ = ["DataSteps", "data_path"]
