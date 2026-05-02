### ~~~ GLOBAL IMPORTS ~~~ ###
from typing import Callable
from pathlib import Path
import json
import os
import re

### ~~~ LOCAL IMPORTS ~~~ ###
from src.ms2.util import data_path, DataSteps, FlattenedExternalData

### ~~~ STATE MANAGEMENT ~~~ ###
# None


def get_paths(root_path: Path) -> list[Path]:
    """
    This function takes a root path as input and
    returns a list of all the file paths in
    that directory.
    Args:
        root_path (Path): The root path to explore.
    Returns:
        list[Path]: A list of file paths in the root directory.
    """
    ### get all the files ###
    files_names: list[str] = os.listdir(root_path)

    ### create a list of paths ###
    files_paths: list[Path] = [root_path / file_name for file_name in files_names]

    return files_paths


def filter_paths(paths: list[Path], extension: str = "json") -> list[Path]:
    """
    This function filters a list of paths based on a specified file extension.
    Args:
        paths (list[Path]): A list of file paths to filter.
        extension (str, optional): The file extension to filter by. Defaults to "json".
    Returns:
        list[Path]: A list of file paths that match the specified extension.
    """
    ### filter the paths based on the extension ###
    filtered_paths: list[Path] = [
        path for path in paths if path.suffix == f".{extension}"
    ]

    return filtered_paths


def train_test_split(
    paths: list[Path], test_regex: str = "test_set"
) -> tuple[list[Path], list[Path]]:
    """
    This function splits a list of paths into training and testing sets based on a
    specified regex pattern.
    Args:
        paths (list[Path]): A list of file paths to split.
        test_regex (str, optional): The regex pattern to identify test set files. Defaults to "test_set".
    Returns:
        tuple[list[Path], list[Path]]: A tuple containing two lists: the first is the
        training set paths, and the second is the testing set paths.
    """
    #### construct the regex ###
    pattern = re.compile(test_regex)

    ### define the filter hueristic ###
    filter_hueristic: Callable = pattern.search

    ### transform the paths into a set for faster lookup ###
    paths_set: set[str] = set(map(str, paths))

    ### split the paths into train and test sets ###
    test_paths_set: set[str] = set(filter(filter_hueristic, paths_set))
    train_paths_set: set[str] = paths_set - test_paths_set

    ### convert the sets back to lists ###
    train_paths: list[Path] = list(map(Path, train_paths_set))
    test_paths: list[Path] = list(map(Path, test_paths_set))

    ### assert same length of train and test sets ###
    assert len(train_paths) == len(test_paths), (
        "Train and test sets must have the same number of files."
    )
    return train_paths, test_paths


def load_json(path: Path) -> dict:
    """
    This function loads a JSON file from a given path and returns its contents as a
    dictionary.
    Args:
        path (Path): The path to the JSON file to load.
    Returns:
        dict: The contents of the JSON file as a dictionary.
    """
    with open(path, "r") as file:
        data = json.load(file)
    return data


def load_data(paths: list[Path]) -> list[FlattenedExternalData]:
    """
    This function does a couple of things:
    1. loads the json files for each path
    2. flatten the json structure
    3. returns a list of dictionaries
    Args:
        paths (list[Path]): A list of file paths to load and process.
    Returns:
        list[dict]: A list of dictionaries containing the processed data from the JSON
        files.
    """
    ### load the json files for each path ###
    raw_objects: list[dict] = list(map(load_json, paths))

    ### define the flattening function ###
    def _flatten_json(raw_object: dict) -> list[FlattenedExternalData]:
        """"""
        ### init some stuff ###
        all_flattened_objects: list[FlattenedExternalData] = []

        ### 1. unpack on `data` key ###
        data: list = raw_object.get("data", [])

        ### 2. delist the list of one item ###
        first_data: dict = {}
        if len(data) == 1:
            first_data = data[0]

        ### 2.5 get the title ###
        title: str = first_data.get("title", "")

        ### 3. unpack on key `paragraphs` ###
        paragraphs: list = first_data.get("paragraphs", [])

        ### 4. for each paragraph, flatten `qas` ###
        for paragraph in paragraphs:
            """
            The structure of the JSON file is as follows:
            {
                "data": [
                    {
                        "title": "some title",
                        "paragraphs": [
                            {
                                "context": "some context",
                                "qas": [
                                    {
                                        "question": "some question",
                                        "answers": [
                                            {
                                                "text": "some answer"
                                            }
                                        ],
                                        "id": "some id"
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
            """
            ## 4.1 get the context and qas ##
            context: str = paragraph.get("context", "")
            qas: list = paragraph.get("qas", [])

            ## 4.2 delist the list of one item ##
            assert len(qas) == 1, (
                "Each paragraph should have exactly one question-answer pair."
            )
            first_qa: dict = qas[0]

            ## 4.3 get the question and answer ##
            question: str = first_qa.get("question", "")

            # warning not clean code #
            answer: str = first_qa.get("answers", [])[0].get("text", "")

            ## 4.4 get the id ##
            id: str = first_qa.get("id", "")

            ## 4.5 construct the flattened object ##
            flattened_object = FlattenedExternalData(
                id=id,
                title=title,
                context=context,
                question=question,
                answer=answer,
            )

            ## 4.6 append the flattened object to the list ##
            all_flattened_objects.append(flattened_object)

        return all_flattened_objects

    ### flatten the json structure ###
    flattened_objects: dict[
        str, list[FlattenedExternalData]
    ] = {  # use the first object to get the title
        objs[0].title: objs for objs in list(map(_flatten_json, raw_objects))
    }

    ### insure all the objects have the same title with a file ###
    flag: bool = all(
        title == obj.title
        for title, list_of_objs in flattened_objects.items()
        for obj in list_of_objs
    )
    assert flag, (
        "All flattened objects must have the same title as their corresponding file."
    )

    ### unpack the flattened objects into a single list ###
    objects: list[FlattenedExternalData] = [
        j for i in flattened_objects.values() for j in i
    ]

    return objects


def load() -> tuple[list[FlattenedExternalData], list[FlattenedExternalData]]:
    """"""
    ### init some vars ###
    root_path: Path = data_path[DataSteps.external]

    ### get all the paths ###
    all_file_paths = get_paths(root_path)

    ### filter the paths based on the extension ###
    all_file_paths = filter_paths(all_file_paths, extension="json")

    ### split the paths into train and test sets ###
    train_paths, test_paths = train_test_split(all_file_paths, test_regex="test_set")

    ### load the data from the train and test paths ###
    train_data: list[FlattenedExternalData] = load_data(train_paths)
    test_data: list[FlattenedExternalData] = load_data(test_paths)

    return train_data, test_data


if __name__ == "__main__":
    ...
#    __   _,_ /_ __,
#  _(_/__(_/_/_)(_/(_
#   _/_
#  (/
