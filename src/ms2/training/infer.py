### ~~~ GLOBAL IMPORT ~~~ ###
import tensorflow as tf

### ~~~ LOCAL IMPORT ~~~ ###
from src.ms2.models.rnn.model import RNNModel

### ~~~ STATE MANAGEMENT ~~~ ###
# None


def greedy_decode_model_a(
    model: RNNModel,
    question_ids: tf.Tensor,
    context_ids: tf.Tensor,
    joint_ids: tf.Tensor,
    bos_id: int,
    eos_id: int,
    max_decode_len: int,
) -> tf.Tensor:
    """
    Greedy decode with the current Model A forward interface.
    Args:
        model: Trained ``RNNModel`` instance.
        question_ids: Question ids ``(B, L_q)``.
        context_ids: Context ids ``(B, L_c)``.
        joint_ids: Joint ids ``(B, L_joint)``.
        bos_id: Decoder start token id.
        eos_id: End-of-sequence token id.
        max_decode_len: Maximum generated length.
    Returns:
        Generated token ids of shape ``(B, max_decode_len)``.
    """
    ### initialize decoder inputs with <bos> ###
    batch_size = tf.shape(question_ids)[0]
    decoder_inputs = tf.fill([batch_size, 1], tf.cast(bos_id, dtype=tf.int32))

    ### collect generated next tokens ###
    generated_steps: list[tf.Tensor] = []
    finished = tf.zeros([batch_size], dtype=tf.bool)

    for _ in range(max_decode_len):
        ### run model with current prefix ###
        logits = model(
            question_ids=question_ids,
            context_ids=context_ids,
            joint_ids=joint_ids,
            decoder_inputs=decoder_inputs,
            training=False,
        )

        ### take argmax from last time step ###
        # dim(last_logits) = (B, V)
        last_logits = logits[:, -1, :]
        next_token = tf.argmax(last_logits, axis=-1, output_type=tf.int32)

        ### if a sequence already finished, keep emitting eos ###
        next_token = tf.where(
            finished,
            tf.fill(tf.shape(next_token), tf.cast(eos_id, dtype=tf.int32)),
            next_token,
        )
        generated_steps.append(next_token)

        ### update finished mask ###
        finished = tf.logical_or(finished, tf.equal(next_token, eos_id))

        ### append next token to decoder prefix ###
        decoder_inputs = tf.concat([decoder_inputs, next_token[:, None]], axis=1)

        ### early stop if all sequences emitted eos ###
        if bool(tf.reduce_all(finished).numpy()):
            break

    ### pad generated list to fixed max_decode_len ###
    if len(generated_steps) == 0:
        return tf.fill([batch_size, max_decode_len], tf.cast(eos_id, dtype=tf.int32))

    generated = tf.stack(generated_steps, axis=1)
    current_len = tf.shape(generated)[1]
    pad_len = tf.maximum(max_decode_len - current_len, 0)
    padding = tf.fill([batch_size, pad_len], tf.cast(eos_id, dtype=tf.int32))
    generated = tf.concat([generated, padding], axis=1)
    return generated[:, :max_decode_len]


def greedy_decode(
    model_name: str,
    model: RNNModel,
    question_ids: tf.Tensor,
    context_ids: tf.Tensor,
    joint_ids: tf.Tensor,
    bos_id: int,
    eos_id: int,
    max_decode_len: int,
) -> tf.Tensor:
    """
    Model-dispatched greedy decode entrypoint.

    This keeps one simple switch so adding Model B later is straightforward.
    """
    if model_name == "a":
        return greedy_decode_model_a(
            model=model,
            question_ids=question_ids,
            context_ids=context_ids,
            joint_ids=joint_ids,
            bos_id=bos_id,
            eos_id=eos_id,
            max_decode_len=max_decode_len,
        )

    raise NotImplementedError("Only model_name='a' decode is implemented right now.")
