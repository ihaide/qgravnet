import tensorflow.keras as keras
import tensorflow as tf

K = keras.backend

from qkeras.quantizers import get_quantizer
from qkeras import QActivation


class QExponential(keras.layers.Layer):
    """
    Implements a quantized element-wise exponential function.
    Formula is given by :math:`y = \exp(\alpha x)`
    Args:
        - alpha: constant in formula as shown above
        - exponential_quantizer: Quantizer confgiuration. Should be of type
          quantized_exp(...)
    """

    def __init__(self, alpha=1.0, exponential_quantizer=None, **kwargs):
        self.alpha = alpha
        if exponential_quantizer is not None:
            self.exponential_quantizer = QActivation(exponential_quantizer)
        else:
            self.exponential_quantizer = None
        super(QExponential, self).__init__(**kwargs)

    def call(self, input):

        x = tf.multiply(self.alpha, input)

        if self.exponential_quantizer is not None:
            output = self.exponential_quantizer(x)
        else:
            output = tf.keras.backend.exp(x)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "alpha": self.alpha,
        }
        base_config = super(QExponential, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_quantization_config(self):
        return {
            "exponential_quantizer": str(self.exponential_quantizer),
            "units": str(self.units),
        }

    def get_quantizers(self):
        return [self.exponential_quantizer]
