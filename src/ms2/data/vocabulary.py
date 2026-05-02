### ~~~ GLOBAL IMPORTS ~~~ ###
from collections import Counter

### ~~~ LOCAL IMPORTS ~~~ ###
from src.ms2.data.normalizer import normalize_arabic_text
from src.ms2.data.tokenizer import tokenize
from src.ms2.util import FlattenedExternalData

### ~~~ STATE MANAGEMENT ~~~ ###
PAD: str = "<PAD>"
UNK: str = "<UNK>"
BOS: str = "<BOS>"
EOS: str = "<EOS>"
SEP: str = "<SEP>"
Q: str = "<Q>"
C: str = "<C>"

SPECIAL_TOKENS: list[str] = [PAD, UNK, BOS, EOS, SEP, Q, C]

SPECIAL_TOKEN_TO_ID: dict[str, int] = {
    PAD: 0,
    UNK: 1,
    BOS: 2,
    EOS: 3,
    SEP: 4,
    Q: 5,
    C: 6,
}


def get_record_tokens(record: FlattenedExternalData) -> list[str]:
    """
    This function gets all model tokens from a single flattened data record.
    Args:
        record (FlattenedExternalData): The flattened data record to tokenize.
    Returns:
        list[str]: A list of tokens from the question, context, and answer.
    """
    ### normalize the text fields ###
    question: str = normalize_arabic_text(record.question)
    context: str = normalize_arabic_text(record.context)
    answer: str = normalize_arabic_text(record.answer)

    ### tokenize the text fields ###
    question_tokens: list[str] = tokenize(question)
    context_tokens: list[str] = tokenize(context)
    answer_tokens: list[str] = tokenize(answer)

    ### add structural tokens for downstream models ###
    tokens: list[str] = [Q, *question_tokens, SEP, C, *context_tokens, SEP, *answer_tokens]

    return tokens


def count_tokens(data: list[FlattenedExternalData]) -> Counter[str]:
    """
    This function counts tokens in a list of flattened training records.
    Args:
        data (list[FlattenedExternalData]): The training data to count tokens from.
    Returns:
        Counter[str]: A counter containing token frequencies.
    """
    ### init the counter ###
    token_counter: Counter[str] = Counter()

    ### count each record tokens ###
    for record in data:
        token_counter.update(get_record_tokens(record))

    return token_counter


def build_vocabulary(
    train_data: list[FlattenedExternalData], min_frequency: int = 1
) -> dict[str, int]:
    """
    This function builds the vocabulary from the training set only.
    Args:
        train_data (list[FlattenedExternalData]): The training data to build from.
        min_frequency (int, optional): The minimum token frequency. Defaults to 1.
    Returns:
        dict[str, int]: A token to id mapping.
    """
    ### count training tokens ###
    token_counter: Counter[str] = count_tokens(train_data)

    ### init vocabulary with special tokens ###
    token_to_id: dict[str, int] = dict(SPECIAL_TOKEN_TO_ID)

    ### sort tokens for deterministic ids ###
    sorted_tokens: list[str] = sorted(token_counter.keys())

    ### add training tokens to the vocabulary ###
    for token in sorted_tokens:
        if token in token_to_id:
            continue
        if token_counter[token] < min_frequency:
            continue
        token_to_id[token] = len(token_to_id)

    return token_to_id


def tokens_to_ids(tokens: list[str], token_to_id: dict[str, int]) -> list[int]:
    """
    This function converts tokens to their ids using the vocabulary.
    Args:
        tokens (list[str]): The tokens to convert.
        token_to_id (dict[str, int]): The token to id mapping.
    Returns:
        list[int]: A list of token ids.
    """
    ### get the unknown token id ###
    unk_id: int = token_to_id[UNK]

    ### convert tokens to ids ###
    token_ids: list[int] = [token_to_id.get(token, unk_id) for token in tokens]

    return token_ids


if __name__ == "__main__":
    ...
#    __   _,_ /_ __,
#  _(_/__(_/_/_)(_/(_
#   _/_
#  (/
