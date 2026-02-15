import tensorflow.keras as keras
import tensorflow as tf

K = keras.backend

from qkeras.quantizers import get_quantizer
from qkeras import QActivation


class QDistanceWeighting(keras.layers.Layer):
    """
    Implements a (not yet) quantized element-wise weighting function.
    Used to apply a distance-based weighting in GravNet.
    Different weighting functions are available:
    - exponential: :math:`y = \exp(\alpha x)` (default, same as QExponential)
    - quadratic: :math:`y = \alpha*(x - \beta)^2 + \epsilon` if x > \beta,
      else :math:`y = \epsilon` where \beta is chosen such that f(x=0) = 1:
      :math:`\beta = \sqrt{(1 - \epsilon) / \alpha}`
    - linear: :math:`y = 1 - \alpha * x` if x < 1/\alpha,
      else :math:`y = \epsilon`

    Args:
        - alpha: constant in formula as shown above
        - epsilon: small constant to avoid returning zero
        - exponential_quantizer: Quantizer configuration. Should be of type
          quantized_exp(...) or None
        - quadratic_quantizer: Quantizer configuration for quadratic weighting.
          Currently not implemented, any non-None value will use the
          quadratic weighting function.
        - linear_quantizer: Quantizer configuration for linear weighting.
          Currently not implemented, any non-None value will use the
          linear weighting function.
        - input_dtype: Data type of the input tensor, default is tf.float32
        - function: Not used, just for showing the type in the config.
    """

    def __init__(
            self,
            alpha=1.0,
            epsilon=0.01,
            max_value=1.0,
            exponential_quantizer=None,
            quadratic_quantizer=None,
            linear_quantizer=None,
            input_dtype=tf.float32,
            function=None,
            **kwargs
        ):
        self.alpha = tf.convert_to_tensor(alpha, dtype=input_dtype)
        self.epsilon = tf.convert_to_tensor(epsilon, dtype=input_dtype)
        self.max_value = tf.convert_to_tensor(max_value, dtype=input_dtype)
        self.beta = K.sqrt((self.max_value - self.epsilon) / self.alpha)
        if exponential_quantizer is not None:
            self.exponential_quantizer = get_quantizer(exponential_quantizer)
            self.function = "exponential"
        else:
            self.exponential_quantizer = None
            self.function = "exponential_no_q"  # Default function is exponential
        if quadratic_quantizer is not None:
            self.quadratic_quantizer = quadratic_quantizer
            self.function = "quadratic"
        else:
            self.quadratic_quantizer = None
        if linear_quantizer is not None:
            self.linear_quantizer = linear_quantizer
            self.function = "linear"
        else:
            self.linear_quantizer = None
        super(QDistanceWeighting, self).__init__(**kwargs)

    def call(self, input):

        if self.exponential_quantizer is not None:
            bits = self.exponential_quantizer.bits
            input_quantizer = get_quantizer(f"quantized_linear({bits+2}, 0)")
            x = input_quantizer(input)
            x = tf.multiply(self.alpha, x)
            output = self.exponential_quantizer(x)
            output = tf.where(input >= 0.5-2**(-(bits+1)), 0.0, output)
        elif self.quadratic_quantizer is not None:
            x = tf.subtract(input, self.beta)
            x = tf.square(x)
            x = tf.multiply(self.alpha, x)
            output = tf.where(
                tf.less(self.beta, input),
                self.epsilon,
                x + self.epsilon
            )
        elif self.linear_quantizer is not None:
            x = tf.multiply(self.alpha, input)
            x = tf.subtract(self.max_value, x)
            x = K.clip(x, self.epsilon, None)
            output = get_quantizer(self.linear_quantizer)(x)
        else:
            x = tf.multiply(self.alpha, input)
            output = tf.keras.backend.exp(x)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "alpha": self.alpha.numpy().item(),
            "epsilon": self.epsilon.numpy().item(),
            "max_value": self.max_value.numpy().item(),
            "function": self.function,
        }
        base_config = super(QDistanceWeighting, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_quantization_config(self):
        if self.exponential_quantizer is not None:
            return {
                "exponential_quantizer": str(self.exponential_quantizer),
                "units": str(self.units),
            }
        if self.quadratic_quantizer is not None:
            return {
                "quadratic_quantizer": str(self.quadratic_quantizer),
                "units": str(self.units),
            }
        if self.linear_quantizer is not None:
            return {
                "linear_quantizer": str(self.linear_quantizer),
                "units": str(self.units),
            }
        return {
            "exponential_quantizer": str(self.exponential_quantizer),
            "units": str(self.units),
        }

    def get_quantizers(self):
        if self.exponential_quantizer is not None:
            return [self.exponential_quantizer]
        if self.quadratic_quantizer is not None:
            return [self.quadratic_quantizer]
        if self.linear_quantizer is not None:
            return [self.linear_quantizer]
        # Default to exponential quantizer if none specified
        return [self.exponential_quantizer]
