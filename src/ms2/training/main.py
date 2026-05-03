### ~~~ GLOBAL IMPORT ~~~ ###
from pathlib import Path
import json
import tensorflow as tf

### ~~~ LOCAL IMPORT ~~~ ###
from src.ms2.models.rnn.model import RNNModel
from src.ms2.training.train import train, build_dataset
from src.ms2.training.infer import greedy_decode
from src.ms2.util import DataSteps, data_path

### ~~~ STATE MANAGEMENT ~~~ ###
# None


def _load_vocab_special_ids() -> tuple[int, int]:
    """
    Load BOS and EOS token ids from the preprocessed vocab file.
    Returns:
        Tuple ``(bos_id, eos_id)``.
    """
    vocab_path: Path = data_path[DataSteps.interim] / "vocab.json"
    with vocab_path.open("r", encoding="utf-8") as file:
        vocab = json.load(file)

    bos_id = int(vocab["special_tokens"]["BOS"])
    eos_id = int(vocab["special_tokens"]["EOS"])
    return bos_id, eos_id


def run() -> None:
    """
    Run a tiny end-to-end dry run for Model A training + inference.
    """
    ### quick training smoke ###
    result = train(model_name="a", batch_size=4, epochs=1, steps_limit_per_epoch=2)
    model: RNNModel = result["model"]

    ### build one batch from test records for decode check ###
    test_path = data_path[DataSteps.interim] / "test_records.json"
    with test_path.open("r", encoding="utf-8") as file:
        test_records: list[dict] = json.load(file)
    test_ds = build_dataset(records=test_records[:4], batch_size=2, shuffle=False)

    features, _ = next(iter(test_ds))
    bos_id, eos_id = _load_vocab_special_ids()

    ### run greedy decoding ###
    generated = greedy_decode(
        model_name="a",
        model=model,
        question_ids=features["question_ids"],
        context_ids=features["context_ids"],
        joint_ids=features["joint_ids"],
        bos_id=bos_id,
        eos_id=eos_id,
        max_decode_len=12,
    )

    print(f"[infer] generated shape={generated.shape}")
    print("[run] status=ok")


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(13)
    run()
