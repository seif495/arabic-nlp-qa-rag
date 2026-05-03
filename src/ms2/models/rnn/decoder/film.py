### ~~~ GLOBAL IMPORT ~~~ ###
import tensorflow as tf

### ~~~ LOCAL IMPORT ~~~ ###
from src.ms2.util import load_pipeline_config, PipelineStep, get_config_value

### ~~~ STATE MANAGEMENT ~~~ ###
# None


class FiLMGenerator(tf.keras.layers.Layer):
    """
    Generate FiLM scale and shift parameters from the encoder summary.
    The encoder summary is mapped to one pair of vectors ``gamma`` and ``beta``.
    These vectors are later broadcast across decoder timesteps and applied after
    each decoder LSTM layer.
    Input shape:
        ``encoder_summary``: ``(B, D_fuse)``
    Output shapes:
        ``gamma``: ``(B, decoder_dim)``
        ``beta``: ``(B, decoder_dim)``
    """

    def __init__(self, **kwargs: object) -> None:
        """
        Initialize the FiLM generator.
        Loads FiLM hyperparameters from the model-definition config and constructs
        a small two-layer MLP. The final layer is initialized so that FiLM starts
        close to the identity transformation: ``gamma ≈ 1`` and ``beta ≈ 0``.
        Args:
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
        hidden_dim: int = get_config_value(config, "rnn_film", "hidden_dim", 128)

        ### store config values needed in call ###
        self.decoder_dim = decoder_dim

        ### hidden projection from encoder summary ###
        self.hidden = tf.keras.layers.Dense(
            hidden_dim,
            activation=tf.keras.activations.gelu,
            name="film_hidden",
        )

        ### final FiLM projection ###
        # This produces both gamma and beta:
        #
        # dim(film_params) = (B, 2 * decoder_dim)
        #
        # The kernel starts very small so the encoder summary has little effect at
        # initialization. The bias is initialized to:
        #
        # gamma bias = 1
        # beta bias  = 0
        self.out = tf.keras.layers.Dense(
            2 * decoder_dim,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
            bias_initializer=self._film_bias_initializer,
            name="film_output",
        )

    def _film_bias_initializer(
        self,
        shape: tuple[int, ...],
        dtype: tf.dtypes.DType | None = None,
    ) -> tf.Tensor:
        """
        Initialize FiLM output bias as ``[ones_for_gamma, zeros_for_beta]``.
        Args:
            shape: Bias shape. Expected to be ``(2 * decoder_dim,)``.
            dtype: Optional TensorFlow dtype.
        Returns:
            Bias tensor of shape ``(2 * decoder_dim,)``.
        """
        del shape

        ### create gamma bias initialized to 1 ###
        gamma_bias = tf.ones((self.decoder_dim,), dtype=dtype)

        ### create beta bias initialized to 0 ###
        beta_bias = tf.zeros((self.decoder_dim,), dtype=dtype)

        ### concatenate into one bias vector ###
        return tf.concat([gamma_bias, beta_bias], axis=0)

    def call(self, encoder_summary: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Generate FiLM parameters from the encoder summary.
        Args:
            encoder_summary: Global encoder summary tensor of shape
                ``(B, D_fuse)``.
        Returns:
            A tuple ``(gamma, beta)`` where both tensors have shape
            ``(B, decoder_dim)``.
        """
        ### compute hidden conditioning features ###
        # dim(encoder_summary) = (B, D_fuse)
        # dim(hidden)          = (B, hidden_dim)
        hidden = self.hidden(encoder_summary)

        ### compute combined FiLM parameters ###
        # dim(film_params) = (B, 2 * decoder_dim)
        film_params = self.out(hidden)

        ### split into scale and shift vectors ###
        # dim(gamma) = (B, decoder_dim)
        # dim(beta)  = (B, decoder_dim)
        gamma, beta = tf.split(film_params, num_or_size_splits=2, axis=-1)

        return gamma, beta
