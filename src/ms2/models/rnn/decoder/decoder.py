### ~~~ GLOBAL IMPORT ~~~ ###
import tensorflow as tf

### ~~~ LOCAL IMPORT ~~~ ###
from src.ms2.models.rnn.embedder import Embedder
from src.ms2.util import load_pipeline_config, PipelineStep, get_config_value

### ~~~ STATE MANAGEMENT ~~~ ###
# None


class Decoder(tf.keras.layers.Layer):
    """
    Two-layer FiLM-conditioned LSTM decoder with a standard vocabulary projection.
    The decoder receives teacher-forced answer-prefix tokens and predicts the next
    token at every timestep. It is conditioned on the encoder summary in two ways:
    - The encoder summary initializes the hidden and cell states of each LSTM.
    - FiLM parameters generated from the encoder summary modulate each LSTM output.
    This version keeps the main no-attention conditioning mechanism while using a
    simple Dense vocabulary head instead of tied output embeddings.
    Input shapes:
        ``decoder_inputs``: ``(B, L_dec)``
        ``encoder_summary``: ``(B, D_fuse)``
        ``gamma``: ``(B, decoder_dim)``
        ``beta``: ``(B, decoder_dim)``
    Output shape:
        ``logits``: ``(B, L_dec, vocab_size)``
    """

    def __init__(self, embedder: Embedder, **kwargs: object) -> None:
        """
        Initialize the decoder.
        Loads decoder hyperparameters from the model-definition config, stores the
        shared token embedder, constructs two LSTM layers, builds encoder-summary
        projections for the initial LSTM states, and defines a Dense vocabulary
        projection.
        Args:
            embedder: Shared token embedding layer used to embed decoder input
                token IDs.
            **kwargs: Additional keyword arguments passed to
                ``tf.keras.layers.Layer``.
        Returns:
            None.
        """
        ### init ###
        super().__init__(**kwargs)

        ### load up the config ###
        config = load_pipeline_config(PipelineStep.model_definition)

        ### get all the config values ###
        decoder_dim: int = get_config_value(config, "rnn_decoder", "decoder_dim", 256)
        dropout_embedding: float = get_config_value(
            config, "rnn_decoder", "dropout_embedding", 0.2
        )
        dropout_rnn: float = get_config_value(config, "rnn_decoder", "dropout_rnn", 0.3)
        use_film: bool = get_config_value(config, "rnn_decoder", "use_film", True)

        ### store config values needed during call ###
        self.decoder_dim = decoder_dim
        self.use_film = use_film

        ### store the shared token embedder ###
        self.embedder = embedder

        ### derive vocab size directly from the shared embedder ###
        # We avoid duplicating vocab size in config to keep one source of truth.
        vocab_size = int(self.embedder.token_embedding.input_dim)

        ### define dropout layers ###
        self.dropout_embedding = tf.keras.layers.Dropout(dropout_embedding)
        self.dropout_rnn = tf.keras.layers.Dropout(dropout_rnn)

        ### define the two decoder LSTM layers ###
        # LSTM 1 receives embedded decoder input tokens.
        #
        # dim(input)  = (B, L_dec, D_embed)
        # dim(output) = (B, L_dec, decoder_dim)
        self.lstm1 = tf.keras.layers.LSTM(
            decoder_dim,
            return_sequences=True,
            return_state=True,
            name="decoder_lstm_1",
        )

        # LSTM 2 receives the hidden sequence from LSTM 1.
        #
        # dim(input)  = (B, L_dec, decoder_dim)
        # dim(output) = (B, L_dec, decoder_dim)
        self.lstm2 = tf.keras.layers.LSTM(
            decoder_dim,
            return_sequences=True,
            return_state=True,
            name="decoder_lstm_2",
        )

        ### define initial-state projections ###
        # Each LSTM layer needs:
        #
        # h_0: initial hidden state
        # c_0: initial cell state
        #
        # Because we have two LSTM layers, we create two h projections and two c
        # projections. Each maps:
        #
        # encoder_summary: (B, D_fuse)
        #      ↓
        # state: (B, decoder_dim)
        self.h_init = [
            tf.keras.layers.Dense(
                decoder_dim,
                activation="tanh",
                name=f"decoder_h_init_{idx}",
            )
            for idx in range(2)
        ]
        self.c_init = [
            tf.keras.layers.Dense(
                decoder_dim,
                activation="tanh",
                name=f"decoder_c_init_{idx}",
            )
            for idx in range(2)
        ]

        ### define output vocabulary projection ###
        # This maps every decoder hidden state to vocabulary logits.
        #
        # dim(input)  = (B, L_dec, decoder_dim)
        # dim(output) = (B, L_dec, vocab_size)
        self.output_projection = tf.keras.layers.Dense(
            vocab_size,
            name="decoder_output_projection",
        )

    def initial_states(self, encoder_summary: tf.Tensor) -> list[list[tf.Tensor]]:
        """
        Generate initial hidden and cell states for both decoder LSTM layers.
        Args:
            encoder_summary: Global encoder summary tensor of shape
                ``(B, D_fuse)``.
        Returns:
            A list containing one ``[h_0, c_0]`` pair per decoder LSTM layer.
            Each state tensor has shape ``(B, decoder_dim)``.
        """
        ### project the encoder summary into LSTM initial states ###
        # The output format matches what Keras LSTM expects for initial_state:
        #
        # initial_state = [h_0, c_0]
        #
        # Since we have two LSTMs, we return:
        #
        # [
        #   [h0_for_lstm1, c0_for_lstm1],
        #   [h0_for_lstm2, c0_for_lstm2],
        # ]
        return [
            [
                self.h_init[idx](encoder_summary),
                self.c_init[idx](encoder_summary),
            ]
            for idx in range(2)
        ]

    def call(
        self,
        decoder_inputs: tf.Tensor,
        encoder_summary: tf.Tensor,
        gamma: tf.Tensor,
        beta: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Decode answer-prefix tokens into vocabulary logits.
        Args:
            decoder_inputs: Integer token ID tensor of shape ``(B, L_dec)``.
            encoder_summary: Global encoder summary tensor of shape
                ``(B, D_fuse)``.
            gamma: FiLM scale tensor of shape ``(B, decoder_dim)``.
            beta: FiLM shift tensor of shape ``(B, decoder_dim)``.
            training: Whether the layer is running in training mode. Controls
                dropout behavior.
        Returns:
            Vocabulary logits of shape ``(B, L_dec, vocab_size)``.
        """
        ### generate initial states from encoder summary ###
        # states[0] is passed to LSTM 1.
        # states[1] is passed to LSTM 2.
        states = self.initial_states(encoder_summary)

        ### embed decoder input tokens ###
        # dim(decoder_inputs) = (B, L_dec)
        # dim(x)              = (B, L_dec, D_embed)
        x = self.embedder(decoder_inputs)

        ### apply embedding dropout ###
        x = self.dropout_embedding(x, training=training)

        ### run decoder LSTM layer 1 ###
        # initial_state=states[0] gives this LSTM encoder-conditioned h_0 and c_0.
        #
        # dim(x) = (B, L_dec, decoder_dim)
        x, *_ = self.lstm1(x, initial_state=states[0], training=training)

        ### apply FiLM after LSTM 1 ###
        # dim(x)     = (B, L_dec, decoder_dim)
        # dim(gamma) = (B, decoder_dim)
        # dim(beta)  = (B, decoder_dim)
        x = self._film(x, gamma, beta)

        ### apply dropout between decoder layers ###
        x = self.dropout_rnn(x, training=training)

        ### run decoder LSTM layer 2 ###
        # dim(x) = (B, L_dec, decoder_dim)
        x, *_ = self.lstm2(x, initial_state=states[1], training=training)

        ### apply FiLM after LSTM 2 ###
        x = self._film(x, gamma, beta)

        ### project to vocabulary logits ###
        # dim(logits) = (B, L_dec, vocab_size)
        logits = self.output_projection(x)

        return logits

    def _film(self, x: tf.Tensor, gamma: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
        """
        Apply FiLM feature-wise affine conditioning.
        FiLM applies:
            ``x' = gamma * x + beta``
        where ``gamma`` and ``beta`` are generated from the encoder summary. Since
        they do not have a time dimension, they are expanded and broadcast across
        all decoder timesteps.
        Args:
            x: Decoder hidden sequence of shape ``(B, L_dec, decoder_dim)``.
            gamma: FiLM scale tensor of shape ``(B, decoder_dim)``.
            beta: FiLM shift tensor of shape ``(B, decoder_dim)``.
        Returns:
            FiLM-conditioned decoder hidden sequence of shape
            ``(B, L_dec, decoder_dim)``.
        """
        ### support the No-FiLM ablation ###
        if not self.use_film:
            return x

        ### broadcast gamma and beta across decoder timesteps ###
        # dim(x)                 = (B, L_dec, decoder_dim)
        # dim(gamma[:, None, :]) = (B, 1, decoder_dim)
        # dim(beta[:, None, :])  = (B, 1, decoder_dim)
        #
        # dim(output)            = (B, L_dec, decoder_dim)
        return x * gamma[:, None, :] + beta[:, None, :]
