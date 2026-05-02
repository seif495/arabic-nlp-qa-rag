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
    "،": "،",
    "؛": "؛",
    "؟": "؟",
}

TATWEEL: str = "ـ"

WHITESPACE_PATTERN = re.compile(r"\s+")
PUNCTUATION_PATTERN = re.compile(r"([،؛؟.!?,;:])")


def normalize_chars(
    text: str,
    normalize_alef: bool = True,
    normalize_ya: bool = True,
    normalize_ta_marbuta: bool = True,
    remove_tatweel: bool = True,
) -> str:
    """
    This function applies light Arabic character normalization.
    Args:
        text (str): The input text to normalize.
    Returns:
        str: The text after normalizing Arabic characters.
    """
    ### init normalized text ###
    normalized_text: str = text

    ### remove tatweel ###
    if remove_tatweel:
        normalized_text = normalized_text.replace(TATWEEL, "")

    ### normalize alef variants ###
    if normalize_alef:
        for old_char, new_char in ALEF_NORMALIZATION.items():
            normalized_text = normalized_text.replace(old_char, new_char)

    ### normalize alef maqsura ###
    if normalize_ya:
        for old_char, new_char in YA_NORMALIZATION.items():
            normalized_text = normalized_text.replace(old_char, new_char)

    ### normalize ta marbuta ###
    if normalize_ta_marbuta:
        normalized_text = normalized_text.replace("ة", "ه")

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


def space_punctuation(text: str) -> str:
    """
    This function adds whitespace around punctuation tokens.
    Args:
        text (str): The input text to normalize.
    Returns:
        str: The text after adding spaces around punctuation.
    """
    ### add spaces around punctuation ###
    normalized_text: str = PUNCTUATION_PATTERN.sub(r" \1 ", text)

    ### clean extra whitespace ###
    normalized_text = normalize_whitespace(normalized_text)

    return normalized_text


def normalize_arabic_text(
    text: str,
    lowercase_english: bool = True,
    normalize_alef: bool = True,
    normalize_ya: bool = True,
    normalize_ta_marbuta: bool = True,
    remove_tatweel: bool = True,
    normalize_spaces: bool = True,
    add_punctuation_spaces: bool = True,
) -> str:
    """
    This function applies light Arabic normalization without aggressive stemming.
    Args:
        text (str): The input text to normalize.
    Returns:
        str: The normalized text.
    """
    ### normalize characters ###
    normalized_text: str = normalize_chars(
        text,
        normalize_alef=normalize_alef,
        normalize_ya=normalize_ya,
        normalize_ta_marbuta=normalize_ta_marbuta,
        remove_tatweel=remove_tatweel,
    )

    ### lowercase english ###
    if lowercase_english:
        normalized_text = normalized_text.lower()

    ### normalize punctuation ###
    normalized_text = normalize_punctuation(normalized_text)

    ### space punctuation ###
    if add_punctuation_spaces:
        normalized_text = space_punctuation(normalized_text)

    ### normalize whitespace ###
    if normalize_spaces:
        normalized_text = normalize_whitespace(normalized_text)

    return normalized_text


if __name__ == "__main__":
    ...
#    __   _,_ /_ __,
#  _(_/__(_/_/_)(_/(_
#   _/_
#  (/
