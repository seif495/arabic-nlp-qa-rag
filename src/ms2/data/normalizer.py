### ~~~ GLOBAL IMPORTS ~~~ ###
import re

### ~~~ LOCAL IMPORTS ~~~ ###
# None

### ~~~ STATE MANAGEMENT ~~~ ###
ALEF_NORMALIZATION: dict[str, str] = {
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
}

YA_NORMALIZATION: dict[str, str] = {
    "ى": "ي",
}

ARABIC_PUNCTUATION_NORMALIZATION: dict[str, str] = {
    "،": ",",
    "؛": ";",
    "؟": "?",
}

TATWEEL: str = "ـ"

WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_chars(text: str) -> str:
    """
    This function applies light Arabic character normalization.
    Args:
        text (str): The input text to normalize.
    Returns:
        str: The text after normalizing Arabic characters.
    """
    ### remove tatweel ###
    normalized_text: str = text.replace(TATWEEL, "")

    ### normalize alef variants ###
    for old_char, new_char in ALEF_NORMALIZATION.items():
        normalized_text = normalized_text.replace(old_char, new_char)

    ### normalize alef maqsura ###
    for old_char, new_char in YA_NORMALIZATION.items():
        normalized_text = normalized_text.replace(old_char, new_char)

    return normalized_text


def normalize_punctuation(text: str) -> str:
    """
    This function normalizes Arabic punctuation while keeping normal tokens unchanged.
    Args:
        text (str): The input text to normalize.
    Returns:
        str: The text after normalizing Arabic punctuation.
    """
    ### normalize arabic punctuation ###
    normalized_text: str = text
    for old_char, new_char in ARABIC_PUNCTUATION_NORMALIZATION.items():
        normalized_text = normalized_text.replace(old_char, new_char)

    return normalized_text


def normalize_whitespace(text: str) -> str:
    """
    This function normalizes whitespace in the input text.
    Args:
        text (str): The input text to normalize.
    Returns:
        str: The text after normalizing whitespace.
    """
    ### normalize whitespace ###
    normalized_text: str = WHITESPACE_PATTERN.sub(" ", text)
    normalized_text = normalized_text.strip()

    return normalized_text


def normalize_arabic_text(text: str) -> str:
    """
    This function applies light Arabic normalization without aggressive stemming.
    Args:
        text (str): The input text to normalize.
    Returns:
        str: The normalized text.
    """
    ### normalize characters ###
    normalized_text: str = normalize_chars(text)

    ### normalize punctuation ###
    normalized_text = normalize_punctuation(normalized_text)

    ### normalize whitespace ###
    normalized_text = normalize_whitespace(normalized_text)

    return normalized_text


if __name__ == "__main__":
    ...
#    __   _,_ /_ __,
#  _(_/__(_/_/_)(_/(_
#   _/_
#  (/
