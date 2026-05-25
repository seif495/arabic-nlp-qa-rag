### ~~~ GLOBAL IMPORT ~~~ ###
import tensorflow as tf

### ~~~ LOCAL IMPORT ~~~ ###
from src.ms2.models.rnn.embedder import get_vocab_size
from src.ms2.models.transformer.embedding import TokenEmbedding
from src.ms2.models.transformer.positional_encoding import PositionalEncoding
from src.ms2.models.transformer.blocks import (
    DecoderBlock,
    EncoderBlock,
    make_causal_mask,
    make_padding_mask,
)
from src.ms2.util import PipelineStep, get_config_value, load_pipeline_config

### ~~~ STATE MANAGEMENT ~~~ ###
# None


class TransformerModel(tf.keras.Model):
    """
    Full vanilla encoder-decoder Transformer for Model B.

    Inputs:
        ``encoder_input_ids``: ``(B, L_enc)``
        ``decoder_inputs``: ``(B, L_dec)``

    Outputs:
        ``logits``: ``(B, L_dec, V)``
    """

    def __init__(self, vocab_size: int | None = None, **kwargs: object) -> None:
        """
        Initialize full Model B vanilla Transformer.

        Args:
            vocab_size: Optional explicit vocabulary size override. If omitted,
                vocab size is derived from the preprocessed vocab file.
            **kwargs: Additional keyword arguments passed to ``tf.keras.Model``.

        Returns:
            None.
        """
        ### init ###
        super().__init__(**kwargs)

        ### load model-definition config ###
        config = load_pipeline_config(PipelineStep.model_definition)

        ### get config values ###
        num_enc_layers = int(
            get_config_value(config, "transformer_encoder", "num_layers", 3)
        )
        num_dec_layers = int(
            get_config_value(config, "transformer_decoder", "num_layers", 3)
        )
        dropout_embedding = float(
            get_config_value(config, "transformer_embedding", "dropout_embedding", 0.1)
        )

        ### resolve vocabulary size ###
        resolved_vocab_size = get_vocab_size() if vocab_size is None else vocab_size

        ### define shared embedding + position path ###
        self.token_embedding = TokenEmbedding(
            vocab_size=resolved_vocab_size, name="tok_emb"
        )
        self.positional_encoding = PositionalEncoding(name="pos_enc")
        self.embedding_dropout = tf.keras.layers.Dropout(dropout_embedding)

        ### define encoder and decoder block stacks ###
        self.encoder_blocks = [
            EncoderBlock(name=f"encoder_block_{idx}") for idx in range(num_enc_layers)
        ]
        self.decoder_blocks = [
            DecoderBlock(name=f"decoder_block_{idx}") for idx in range(num_dec_layers)
        ]

        ### define final normalization and vocab projection ###
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.output_projection = tf.keras.layers.Dense(
            resolved_vocab_size,
            name="transformer_output_projection",
        )

    def call(
        self,
        encoder_input_ids: tf.Tensor,
        decoder_inputs: tf.Tensor,
        training: bool = False,
        return_details: bool = False,
    ) -> tf.Tensor | dict[str, tf.Tensor]:
        """
        Run full transformer forward pass.

        Args:
            encoder_input_ids: Encoder token IDs of shape ``(B, L_enc)``.
            decoder_inputs: Teacher-forced decoder prefix IDs of shape
                ``(B, L_dec)``.
            training: Whether the model is running in training mode.
            return_details: Whether to return auxiliary tensors.

        Returns:
            Either logits tensor of shape ``(B, L_dec, V)`` or a dictionary
            containing logits and auxiliary tensors.
        """
        ### build masks ###
        # dim(enc_pad) = (B, 1, L_enc)
        # dim(dec_pad) = (B, 1, L_dec)
        # dim(causal)  = (1, L_dec, L_dec)
        # dim(self_mask) = (B, L_dec, L_dec)
        enc_pad = make_padding_mask(encoder_input_ids)
        dec_pad = make_padding_mask(decoder_inputs)
        dec_len = tf.shape(decoder_inputs)[1]
        causal = make_causal_mask(dec_len)
        self_mask = tf.logical_and(dec_pad, causal)

        ### encoder embedding path ###
        # dim(enc) = (B, L_enc, d_model)
        enc = self.token_embedding(encoder_input_ids)
        enc = self.positional_encoding(enc, training=training)
        enc = self.embedding_dropout(enc, training=training)

        ### encoder blocks ###
        # dim(enc) stays (B, L_enc, d_model)
        for block in self.encoder_blocks:
            enc = block(enc, padding_mask=enc_pad, training=training)

        ### decoder embedding path ###
        # dim(dec) = (B, L_dec, d_model)
        dec = self.token_embedding(decoder_inputs)
        dec = self.positional_encoding(dec, training=training)
        dec = self.embedding_dropout(dec, training=training)

        ### decoder blocks ###
        # dim(dec) stays (B, L_dec, d_model)
        for block in self.decoder_blocks:
            dec = block(
                dec,
                memory=enc,
                self_mask=self_mask,
                memory_mask=enc_pad,
                training=training,
            )

        ### final projection to vocabulary logits ###
        # dim(logits) = (B, L_dec, V)
        dec = self.final_norm(dec)
        logits = self.output_projection(dec)

        if return_details:
            return {
                "logits": logits,
                "encoder_output": enc,
                "encoder_padding_mask": enc_pad,
            }
        return logits


def main() -> None:
    """Dry-run smoke test for Model B dimensions."""
    vocab_size = 128
    batch_size = 2
    l_enc = 24
    l_dec = 8

    model = TransformerModel(vocab_size=vocab_size, name="transformer_model")
    enc_in = tf.keras.Input(shape=(l_enc,), dtype=tf.int32, name="encoder_input_ids")
    dec_in = tf.keras.Input(shape=(l_dec,), dtype=tf.int32, name="decoder_inputs")
    logits = model(encoder_input_ids=enc_in, decoder_inputs=dec_in)
    graph_model = tf.keras.Model(inputs=[enc_in, dec_in], outputs=logits)
    graph_model.compile(optimizer=tf.keras.optimizers.Adam())

    encoder_input_ids = tf.random.uniform(
        shape=(batch_size, l_enc), minval=1, maxval=vocab_size, dtype=tf.int32
    )
    decoder_inputs = tf.random.uniform(
        shape=(batch_size, l_dec), minval=1, maxval=vocab_size, dtype=tf.int32
    )
    out = graph_model([encoder_input_ids, decoder_inputs], training=False)
    print(f"logits shape: {out.shape}")
    print("transformer dry-run: ok")


if __name__ == "__main__":
    main()
