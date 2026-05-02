### ~~~ GLOBAL IMPORTS ~~~ ###
from collections import Counter
from pathlib import Path
import json
import re

### ~~~ LOCAL IMPORTS ~~~ ###
from src.ms2.data.loader import load
from src.ms2.data.normalizer import normalize_arabic_text
from src.ms2.data.tokenizer import tokenize_with_offsets
from src.ms2.data.vocabulary import BOS, C, EOS, PAD, Q, SEP, SPECIAL_TOKEN_TO_ID, UNK
from src.ms2.util import (
    DataSteps,
    FlattenedExternalData,
    data_path,
    congif_path,
    PipelineSteps,
)

### ~~~ STATE MANAGEMENT ~~~ ###
DEFAULT_CONFIG_PATH: Path = congif_path[PipelineSteps.data_preprocessing]

ARABIC_PATTERN = re.compile(
    r"^[\u0621-\u063A\u0641-\u064A\u064B-\u065F\u0670\u0671-\u06D3\u06FA-\u06FC]+$"
)
ENGLISH_PATTERN = re.compile(r"^[A-Za-z]+$")
NUMBER_PATTERN = re.compile(r"^\d+(?:\.\d+)?$")
PUNCTUATION_PATTERN = re.compile(r"^[^\w\s]$")


def parse_config_value(value: str) -> object:
    """
    This function parses a simple YAML scalar value.
    Args:
        value (str): The raw value from the config file.
    Returns:
        object: The parsed config value.
    """
    ### clean the value ###
    cleaned_value: str = value.strip()

    ### parse booleans ###
    if cleaned_value == "true":
        return True
    if cleaned_value == "false":
        return False

    ### parse numbers ###
    if cleaned_value.isdigit():
        return int(cleaned_value)
    try:
        return float(cleaned_value)
    except ValueError:
        pass

    ### parse strings ###
    return cleaned_value.strip('"')


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """
    This function loads the preprocessing config from a simple YAML file.
    Args:
        config_path (Path, optional): The config path. Defaults to DEFAULT_CONFIG_PATH.
    Returns:
        dict: The loaded config dictionary.
    """
    ### init config objects ###
    config: dict = {}
    current_section: str = ""

    ### parse the config file ###
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line: str = raw_line.split("#", maxsplit=1)[0].rstrip()
        if not line:
            continue

        ### parse section names ###
        if not raw_line.startswith(" "):
            current_section = line.replace(":", "")
            config[current_section] = {}
            continue

        ### parse section values ###
        key, value = line.strip().split(":", maxsplit=1)
        config[current_section][key] = parse_config_value(value)

    return config


def get_config_value(config: dict, section: str, key: str, default: object) -> object:
    """
    This function gets a config value with a default fallback.
    Args:
        config (dict): The loaded config dictionary.
        section (str): The config section name.
        key (str): The config key name.
        default (object): The default value.
    Returns:
        object: The config value.
    """
    ### get config value ###
    value: object = config.get(section, {}).get(key, default)

    return value


def normalize_field(text: str, config: dict) -> str:
    """
    This function normalizes one text field using the config.
    Args:
        text (str): The raw input text.
        config (dict): The preprocessing config.
    Returns:
        str: The normalized text.
    """
    ### normalize the text ###
    normalized_text: str = normalize_arabic_text(
        text,
        lowercase_english=bool(
            get_config_value(config, "normalization", "lowercase_english", True)
        ),
        normalize_alef=bool(
            get_config_value(config, "normalization", "normalize_alef", True)
        ),
        normalize_ya=bool(
            get_config_value(config, "normalization", "normalize_ya", True)
        ),
        normalize_ta_marbuta=bool(
            get_config_value(config, "normalization", "normalize_ta_marbuta", True)
        ),
        remove_tatweel=bool(
            get_config_value(config, "normalization", "remove_tatweel", True)
        ),
        normalize_spaces=bool(
            get_config_value(config, "normalization", "normalize_whitespace", True)
        ),
        add_punctuation_spaces=bool(
            get_config_value(config, "normalization", "space_punctuation", True)
        ),
    )

    return normalized_text


def classify_token(token: str, language_ids: dict) -> int:
    """
    This function classifies a token into a language/channel id.
    Args:
        token (str): The token to classify.
        language_ids (dict): The configured language id mapping.
    Returns:
        int: The token language id.
    """
    ### classify special tokens ###
    if token in SPECIAL_TOKEN_TO_ID:
        return int(language_ids["special"])

    ### classify normal tokens ###
    if ARABIC_PATTERN.match(token):
        return int(language_ids["arabic"])
    if ENGLISH_PATTERN.match(token):
        return int(language_ids["english"])
    if NUMBER_PATTERN.match(token):
        return int(language_ids["number"])
    if PUNCTUATION_PATTERN.match(token):
        return int(language_ids["punctuation"])

    return int(language_ids["other"])


def get_language_ids(config: dict) -> dict:
    """
    This function gets language ids from the preprocessing config.
    Args:
        config (dict): The preprocessing config.
    Returns:
        dict: The configured language ids.
    """
    ### get language id section ###
    language_ids: dict = config.get("language_ids", {})

    return language_ids


def build_vocab_from_records(records: list[dict], min_frequency: int) -> dict[str, int]:
    """
    This function builds the vocabulary from training records only.
    Args:
        records (list[dict]): The train records after tokenization.
        min_frequency (int): The minimum token frequency.
    Returns:
        dict[str, int]: A token to id mapping.
    """
    ### init counter ###
    token_counter: Counter[str] = Counter()

    ### count train tokens ###
    for record in records:
        token_counter.update(record["tokens"]["question"])
        token_counter.update(record["tokens"]["context"])
        token_counter.update(record["tokens"]["answer"])

    ### init vocabulary with special tokens ###
    token_to_id: dict[str, int] = dict(SPECIAL_TOKEN_TO_ID)

    ### add tokens in deterministic order ###
    for token in sorted(token_counter.keys()):
        if token in token_to_id:
            continue
        if token_counter[token] < min_frequency:
            continue
        token_to_id[token] = len(token_to_id)

    return token_to_id


def tokens_to_ids(tokens: list[str], token_to_id: dict[str, int]) -> list[int]:
    """
    This function converts tokens to ids.
    Args:
        tokens (list[str]): The input tokens.
        token_to_id (dict[str, int]): The token to id mapping.
    Returns:
        list[int]: The token ids.
    """
    ### convert tokens to ids ###
    ids: list[int] = [token_to_id.get(token, token_to_id[UNK]) for token in tokens]

    return ids


def token_overlap(first_tokens: list[str], second_tokens: list[str]) -> float:
    """
    This function computes token set overlap.
    Args:
        first_tokens (list[str]): The first token list.
        second_tokens (list[str]): The second token list.
    Returns:
        float: The token overlap score.
    """
    ### convert tokens to sets ###
    first_set: set[str] = set(first_tokens)
    second_set: set[str] = set(second_tokens)

    ### compute overlap ###
    overlap: float = len(first_set & second_set) / max(len(first_set), 1)

    return round(overlap, 4)


def get_exact_token_span(
    answer_tokens: list[str], context_tokens: list[str]
) -> tuple[int, int] | None:
    """
    This function finds an exact answer token span in context tokens.
    Args:
        answer_tokens (list[str]): The normalized answer tokens.
        context_tokens (list[str]): The normalized context tokens.
    Returns:
        tuple[int, int] | None: The inclusive token span or None.
    """
    ### handle empty answers ###
    if not answer_tokens:
        return None

    ### search exact token windows ###
    answer_length: int = len(answer_tokens)
    for start in range(len(context_tokens) - answer_length + 1):
        end: int = start + answer_length
        if context_tokens[start:end] == answer_tokens:
            return start, end - 1

    return None


def score_window(answer_tokens: list[str], window_tokens: list[str]) -> float:
    """
    This function computes token F1 between answer tokens and a context window.
    Args:
        answer_tokens (list[str]): The answer tokens.
        window_tokens (list[str]): The context window tokens.
    Returns:
        float: The token F1 score.
    """
    ### count token overlap ###
    answer_counter: Counter[str] = Counter(answer_tokens)
    window_counter: Counter[str] = Counter(window_tokens)
    overlap_count: int = sum((answer_counter & window_counter).values())

    ### handle empty overlap ###
    if overlap_count == 0:
        return 0.0

    ### compute f1 ###
    precision: float = overlap_count / max(len(window_tokens), 1)
    recall: float = overlap_count / max(len(answer_tokens), 1)
    f1: float = 2 * precision * recall / max(precision + recall, 1e-8)

    return f1


def get_token_f1_span(
    answer_tokens: list[str], context_tokens: list[str]
) -> tuple[int, int, float]:
    """
    This function finds the best token-F1 answer span in the context.
    Args:
        answer_tokens (list[str]): The answer tokens.
        context_tokens (list[str]): The context tokens.
    Returns:
        tuple[int, int, float]: The inclusive token span and alignment score.
    """
    ### init best span ###
    best_start: int = -1
    best_end: int = -1
    best_score: float = 0.0
    max_window_size: int = max(len(answer_tokens) + 3, 1)

    ### search context windows ###
    for start in range(len(context_tokens)):
        for end in range(start, min(len(context_tokens), start + max_window_size)):
            window_tokens: list[str] = context_tokens[start : end + 1]
            score: float = score_window(answer_tokens, window_tokens)
            if score > best_score:
                best_start = start
                best_end = end
                best_score = score

    return best_start, best_end, round(best_score, 4)


def build_span_target(record: dict, config: dict) -> dict:
    """
    This function builds span objective fields for one record.
    Args:
        record (dict): The preprocessed record.
        config (dict): The preprocessing config.
    Returns:
        dict: The span target fields.
    """
    ### get record data ###
    answer_tokens: list[str] = record["tokens"]["answer"]
    context_tokens: list[str] = record["tokens"]["context"]
    question_tokens: list[str] = record["tokens"]["question"]
    ### try exact match first ###
    exact_span = get_exact_token_span(answer_tokens, context_tokens)
    is_exact_match: bool = exact_span is not None

    if exact_span is not None:
        answer_start, answer_end = exact_span
        alignment_score: float = 1.0
    else:
        answer_start, answer_end, alignment_score = get_token_f1_span(
            answer_tokens, context_tokens
        )

    ### convert context span to joint span ###
    context_start_in_joint: int = 2 + len(question_tokens) + 1 + 1
    answer_start_in_joint: int = -1
    answer_end_in_joint: int = -1
    if answer_start != -1:
        answer_start_in_joint = context_start_in_joint + answer_start
        answer_end_in_joint = context_start_in_joint + answer_end

    ### compute weak alignment flag ###
    weak_threshold: float = float(
        get_config_value(config, "objectives", "weak_alignment_threshold", 0.5)
    )
    is_weak_alignment: bool = alignment_score < weak_threshold

    ### construct span target ###
    span_target: dict = {
        "answer_start": answer_start,
        "answer_end": answer_end,
        "answer_start_in_joint": answer_start_in_joint,
        "answer_end_in_joint": answer_end_in_joint,
        "alignment_method": get_config_value(
            config, "objectives", "alignment_method", "token_f1"
        ),
        "alignment_score": alignment_score,
        "is_exact_match": is_exact_match,
        "is_weak_alignment": is_weak_alignment,
        "valid_for_span_training": answer_start != -1 and alignment_score > 0,
    }

    return span_target


def build_base_record(record: FlattenedExternalData, split: str, config: dict) -> dict:
    """
    This function builds one normalized and tokenized record.
    Args:
        record (FlattenedExternalData): The source record.
        split (str): The dataset split name.
        config (dict): The preprocessing config.
    Returns:
        dict: The base preprocessed record without ids.
    """
    ### create raw fields ###
    raw: dict[str, str] = {
        "question": record.question,
        "context": record.context,
        "answer": record.answer,
    }

    ### create normalized fields ###
    normalized: dict[str, str] = {
        key: normalize_field(value, config) for key, value in raw.items()
    }

    ### tokenize raw fields and keep raw offsets ###
    question_tokens, question_offsets = tokenize_raw_with_normalized_tokens(
        raw["question"], config
    )
    context_tokens, context_offsets = tokenize_raw_with_normalized_tokens(
        raw["context"], config
    )
    answer_tokens, answer_offsets = tokenize_raw_with_normalized_tokens(
        raw["answer"], config
    )

    ### construct record object ###
    preprocessed_record: dict = {
        "id": record.id,
        "title": record.title,
        "split": split,
        "raw": raw,
        "normalized": normalized,
        "tokens": {
            "question": question_tokens,
            "context": context_tokens,
            "answer": answer_tokens,
        },
        "offsets": {
            "question": question_offsets,
            "context": context_offsets,
            "answer": answer_offsets,
        },
    }

    return preprocessed_record


def tokenize_raw_with_normalized_tokens(
    raw_text: str, config: dict
) -> tuple[list[str], list[list[int]]]:
    """
    This function tokenizes raw text and normalizes each token.
    Args:
        raw_text (str): The raw input text.
        config (dict): The preprocessing config.
    Returns:
        tuple[list[str], list[list[int]]]: Normalized tokens and raw offsets.
    """
    ### tokenize raw text for raw offsets ###
    raw_tokens, raw_offsets = tokenize_with_offsets(raw_text)

    ### normalize each token while keeping the raw offsets ###
    normalized_tokens: list[str] = []
    normalized_offsets: list[list[int]] = []
    for token, offset in zip(raw_tokens, raw_offsets):
        normalized_token: str = normalize_field(token, config)
        if not normalized_token:
            continue
        normalized_tokens.append(normalized_token)
        normalized_offsets.append(offset)

    return normalized_tokens, normalized_offsets


def add_ids_and_targets(
    record: dict, token_to_id: dict[str, int], config: dict
) -> dict:
    """
    This function adds ids, features, targets, masks, and diagnostics to a record.
    Args:
        record (dict): The base preprocessed record.
        token_to_id (dict[str, int]): The token to id mapping.
        config (dict): The preprocessing config.
    Returns:
        dict: The full model-ready preprocessing record.
    """
    ### get tokens ###
    question_tokens: list[str] = record["tokens"]["question"]
    context_tokens: list[str] = record["tokens"]["context"]
    answer_tokens: list[str] = record["tokens"]["answer"]
    joint_tokens: list[str] = [
        BOS,
        Q,
        *question_tokens,
        SEP,
        C,
        *context_tokens,
        EOS,
    ]

    ### convert tokens to ids ###
    question_ids: list[int] = tokens_to_ids(question_tokens, token_to_id)
    context_ids: list[int] = tokens_to_ids(context_tokens, token_to_id)
    answer_ids: list[int] = tokens_to_ids(answer_tokens, token_to_id)
    joint_ids: list[int] = tokens_to_ids(joint_tokens, token_to_id)

    ### add ids ###
    record["ids"] = {
        "question": question_ids,
        "context": context_ids,
        "answer": answer_ids,
        "joint": joint_ids,
    }

    ### add feature channels ###
    language_ids: dict = get_language_ids(config)
    record["features"] = {
        "joint_segment_ids": [0] * (2 + len(question_tokens) + 1)
        + [1] * (1 + len(context_tokens) + 1),
        "joint_language_ids": [
            classify_token(token, language_ids) for token in joint_tokens
        ],
        "joint_position_ids": list(range(len(joint_tokens))),
    }

    ### add span target ###
    record["span_target"] = build_span_target(record, config)

    ### add generation target ###
    decoder_tokens_input: list[str] = [BOS, *answer_tokens]
    decoder_tokens_target: list[str] = [*answer_tokens, EOS]
    record["generation_target"] = {
        "decoder_input_ids": tokens_to_ids(decoder_tokens_input, token_to_id),
        "decoder_target_ids": tokens_to_ids(decoder_tokens_target, token_to_id),
        "decoder_tokens_input": decoder_tokens_input,
        "decoder_tokens_target": decoder_tokens_target,
        "valid_for_generation_training": len(answer_tokens) > 0,
    }

    ### add masks ###
    record["masks"] = {
        "question_mask": [1] * len(question_ids),
        "context_mask": [1] * len(context_ids),
        "joint_mask": [1] * len(joint_ids),
        "decoder_mask": [1] * len(decoder_tokens_input),
    }

    ### add diagnostics ###
    english_tokens: list[str] = [
        token for token in joint_tokens if ENGLISH_PATTERN.match(token)
    ]
    record["diagnostics"] = {
        "question_len": len(question_tokens),
        "context_len": len(context_tokens),
        "answer_len": len(answer_tokens),
        "has_english": len(english_tokens) > 0,
        "english_tokens": sorted(set(english_tokens)),
        "answer_is_substring": record["normalized"]["answer"]
        in record["normalized"]["context"],
        "question_context_overlap": token_overlap(question_tokens, context_tokens),
        "answer_context_overlap": token_overlap(answer_tokens, context_tokens),
    }

    return record


def save_json(path: Path, data: object) -> None:
    """
    This function saves an object as JSON.
    Args:
        path (Path): The output path.
        data (object): The data to save.
    Returns:
        None
    """
    ### save the json file ###
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def build_vocab_object(token_to_id: dict[str, int]) -> dict:
    """
    This function builds the saved vocabulary object.
    Args:
        token_to_id (dict[str, int]): The token to id mapping.
    Returns:
        dict: The vocabulary object to save.
    """
    ### build special token mapping ###
    special_tokens: dict[str, int] = {
        "PAD": token_to_id[PAD],
        "UNK": token_to_id[UNK],
        "BOS": token_to_id[BOS],
        "EOS": token_to_id[EOS],
        "SEP": token_to_id[SEP],
        "Q": token_to_id[Q],
        "C": token_to_id[C],
    }

    ### build id to token mapping ###
    id_to_token: dict[str, str] = {
        str(token_id): token for token, token_id in token_to_id.items()
    }

    return {
        "special_tokens": special_tokens,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
    }


def build_stats(
    train_records: list[dict], test_records: list[dict], config: dict
) -> dict:
    """
    This function builds preprocessing summary statistics.
    Args:
        train_records (list[dict]): The train records.
        test_records (list[dict]): The test records.
        config (dict): The preprocessing config.
    Returns:
        dict: The preprocessing stats.
    """
    ### collect all records ###
    all_records: list[dict] = [*train_records, *test_records]

    ### count token channels ###
    language_ids: dict = get_language_ids(config)
    reverse_language_ids: dict[int, str] = {
        int(value): key for key, value in language_ids.items()
    }
    language_counter: Counter[str] = Counter()
    for record in all_records:
        for language_id in record["features"]["joint_language_ids"]:
            language_counter[reverse_language_ids[int(language_id)]] += 1

    ### build stats ###
    stats: dict = {
        "train_records": len(train_records),
        "test_records": len(test_records),
        "vocab_min_frequency": get_config_value(
            config, "vocabulary", "min_frequency", 1
        ),
        "language_counts": dict(language_counter),
        "span_valid_records": sum(
            1
            for record in all_records
            if record["span_target"]["valid_for_span_training"]
        ),
        "generation_valid_records": sum(
            1
            for record in all_records
            if record["generation_target"]["valid_for_generation_training"]
        ),
    }

    return stats


def preprocess(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """
    This function runs the full MS2 preprocessing pipeline.
    Args:
        config_path (Path, optional): The preprocessing config path. Defaults to DEFAULT_CONFIG_PATH.
    Returns:
        dict: The preprocessing run summary.
    """
    ### load config and data ###
    config: dict = load_config(config_path)
    train_data, test_data = load()

    ### build base records ###
    train_records: list[dict] = [
        build_base_record(record, "train", config) for record in train_data
    ]
    test_records: list[dict] = [
        build_base_record(record, "test", config) for record in test_data
    ]

    ### build train-only vocabulary ###
    min_frequency: int = int(get_config_value(config, "vocabulary", "min_frequency", 1))
    token_to_id: dict[str, int] = build_vocab_from_records(train_records, min_frequency)

    ### finish records with ids and targets ###
    train_records = [
        add_ids_and_targets(record, token_to_id, config) for record in train_records
    ]
    test_records = [
        add_ids_and_targets(record, token_to_id, config) for record in test_records
    ]

    ### prepare output paths ###
    output_dir: Path = Path(
        str(
            get_config_value(
                config, "paths", "output_dir", data_path[DataSteps.interim]
            )
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path: Path = output_dir / str(
        get_config_value(config, "paths", "train_records", "train_records.json")
    )
    test_path: Path = output_dir / str(
        get_config_value(config, "paths", "test_records", "test_records.json")
    )
    vocab_path: Path = output_dir / str(
        get_config_value(config, "paths", "vocab", "vocab.json")
    )
    stats_path: Path = output_dir / str(
        get_config_value(config, "paths", "stats", "preprocessing_stats.json")
    )

    ### build saved objects ###
    vocab_object: dict = build_vocab_object(token_to_id)
    stats: dict = build_stats(train_records, test_records, config)
    stats["vocab_size"] = len(token_to_id)

    ### save outputs ###
    save_json(train_path, train_records)
    save_json(test_path, test_records)
    save_json(vocab_path, vocab_object)
    save_json(stats_path, stats)

    ### return summary ###
    summary: dict = {
        "train_records": len(train_records),
        "test_records": len(test_records),
        "vocab_size": len(token_to_id),
        "output_dir": str(output_dir),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "vocab_path": str(vocab_path),
        "stats_path": str(stats_path),
    }

    return summary


if __name__ == "__main__":
    print(preprocess())
#    __   _,_ /_ __,
#  _(_/__(_/_/_)(_/(_
#   _/_
#  (/
