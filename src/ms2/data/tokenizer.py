### ~~~ GLOBAL IMPORTS ~~~ ###
import re

### ~~~ LOCAL IMPORTS ~~~ ###
# None

### ~~~ STATE MANAGEMENT ~~~ ###
TOKEN_PATTERN = re.compile(r"[\u0600-\u06FF]+|[A-Za-z]+|\d+(?:\.\d+)?|[^\w\s]")


def tokenize(text: str) -> list[str]:
    """
    This function tokenizes mixed Arabic-English text.
    Args:
        text (str): The input text to tokenize.
    Returns:
        list[str]: A list of tokens extracted from the input text.
    """
    ### split words, numbers, and punctuation ###
    tokens: list[str] = TOKEN_PATTERN.findall(text)

    return tokens


if __name__ == "__main__":
    ...
#    __   _,_ /_ __,
#  _(_/__(_/_/_)(_/(_
#   _/_
#  (/
