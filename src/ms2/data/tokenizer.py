### ~~~ GLOBAL IMPORTS ~~~ ###
import re

### ~~~ LOCAL IMPORTS ~~~ ###
# None

### ~~~ STATE MANAGEMENT ~~~ ###
ARABIC_WORD_REGEX: str = (
    r"[\u0621-\u063A\u0641-\u064A\u064B-\u065F\u0670\u0671-\u06D3\u06FA-\u06FC]+"
)
TOKEN_PATTERN = re.compile(rf"{ARABIC_WORD_REGEX}|[A-Za-z]+|\d+(?:\.\d+)?|[^\w\s]")
ARABIC_WORD_PATTERN = re.compile(rf"^{ARABIC_WORD_REGEX}$")
ARABIC_PREFIXES: list[str] = ["و", "ف", "ب", "ل"]


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


def tokenize_with_offsets(text: str) -> tuple[list[str], list[list[int]]]:
    """
    This function tokenizes mixed Arabic-English text and returns token offsets.
    Args:
        text (str): The input text to tokenize.
    Returns:
        tuple[list[str], list[list[int]]]: The tokens and their start/end offsets.
    """
    ### split words, numbers, and punctuation with offsets ###
    matches: list[re.Match] = list(TOKEN_PATTERN.finditer(text))

    ### extract tokens and offsets ###
    tokens: list[str] = [match.group() for match in matches]
    offsets: list[list[int]] = [[match.start(), match.end()] for match in matches]

    return tokens, offsets


def tokenize_words(text: str) -> list[str]:
    """
    This function applies word-level tokenization.
    Args:
        text (str): The input text to tokenize.
    Returns:
        list[str]: A list of word-level tokens.
    """
    ### use the default tokenizer ###
    tokens: list[str] = tokenize(text)

    return tokens


def tokenize_chars(token: str) -> list[str]:
    """
    This function applies character-level tokenization to one token.
    Args:
        token (str): The input token to split into characters.
    Returns:
        list[str]: A list of character tokens.
    """
    ### split token into characters ###
    chars: list[str] = list(token)

    return chars


def tokenize_text_chars(text: str) -> list[list[str]]:
    """
    This function applies character-level tokenization to text tokens.
    Args:
        text (str): The input text to tokenize.
    Returns:
        list[list[str]]: A list of character tokens for each word token.
    """
    ### get word tokens first ###
    word_tokens: list[str] = tokenize_words(text)

    ### split each word token into characters ###
    char_tokens: list[list[str]] = [tokenize_chars(token) for token in word_tokens]

    return char_tokens


def tokenize_arabic_subword(token: str) -> list[str]:
    """
    This function applies simple Arabic pseudo-subword tokenization.
    Args:
        token (str): The input token to split.
    Returns:
        list[str]: A list of pseudo-subword tokens.
    """
    ### keep non-arabic tokens unchanged ###
    if not ARABIC_WORD_PATTERN.match(token):
        return [token]

    ### split prefix plus definite article ###
    for prefix in ARABIC_PREFIXES:
        article_prefix: str = f"{prefix}ال"
        if token.startswith(article_prefix) and len(token) > len(article_prefix) + 1:
            return [prefix, "ال", token[len(article_prefix) :]]

    ### split definite article ###
    if token.startswith("ال") and len(token) > 3:
        return ["ال", token[2:]]

    ### split single-letter prefix ###
    for prefix in ARABIC_PREFIXES:
        if token.startswith(prefix) and len(token) > 2:
            return [prefix, token[1:]]

    return [token]


def tokenize_text_subwords(text: str) -> list[str]:
    """
    This function applies optional pseudo-subword tokenization to text.
    Args:
        text (str): The input text to tokenize.
    Returns:
        list[str]: A list of pseudo-subword tokens.
    """
    ### get word tokens first ###
    word_tokens: list[str] = tokenize_words(text)

    ### split arabic tokens into pseudo-subwords ###
    subword_tokens: list[str] = [
        subword for token in word_tokens for subword in tokenize_arabic_subword(token)
    ]

    return subword_tokens


def tokenize_hybrid(text: str, use_subwords: bool = False) -> dict[str, list]:
    """
    This function applies hybrid tokenization with configurable subwords.
    Args:
        text (str): The input text to tokenize.
        use_subwords (bool, optional): Whether to include pseudo-subwords. Defaults to False.
    Returns:
        dict[str, list]: A dictionary containing the enabled tokenization levels.
    """
    ### create the default hybrid output ###
    tokens: dict[str, list] = {
        "words": tokenize_words(text),
        "chars": tokenize_text_chars(text),
    }

    ### optionally add pseudo-subwords ###
    if use_subwords:
        tokens["subwords"] = tokenize_text_subwords(text)

    return tokens


if __name__ == "__main__":
    ...
#    __   _,_ /_ __,
#  _(_/__(_/_/_)(_/(_
#   _/_
#  (/
