### ~~~ GLOBAL IMPORTS ~~~ ###
from dataclasses import dataclass

### ~~~ LOCAL IMPORTS ~~~ ###
# None


### ~~~ STATE MANAGEMENT ~~~ ###
@dataclass
class FlattenedExternalData:
    """"""

    id: str
    title: str
    context: str
    question: str
    answer: str
