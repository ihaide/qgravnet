import tensorflow.keras as keras
import tensorflow as tf

from qkeras.quantizers import get_quantizer


class QGlobalAverageReduce1D(keras.layers.Layer):
    """
    This Layer Reduces the innermost dimension of an arbitrary input tensor
    by calculating the arithmetic average on it.
    Args:
     - kernel_quantizer: Quantization of the output value

     - divider_quantizer: Quantization of the divider value
    Shapes:
     - **:input:**
       Tensor of rank > 0
       n: Number of active inputs. Muste be a tensor of rank 0.
     - **:output:**
       Input tensor rank-1
    """

    def __init__(self, kernel_quantizer=None, divider_quantizer=None, **kwargs):
        if kernel_quantizer is not None:
            self.kernel_quantizer = get_quantizer(kernel_quantizer)
        else:
            self.kernel_quantizer = None
        if divider_quantizer is not None:
            self.divider_quantizer = get_quantizer(divider_quantizer)
        else:
            self.divider_quantizer = None
        super(QGlobalAverageReduce1D, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) > 1
        self.n = tf.cast(input_shape[-1], tf.float32)
        super(QGlobalAverageReduce1D, self).build(input_shape)

    def call(self, input):
        # Separation of addition and division to account for loss in precision

        if self.divider_quantizer is not None:
            n_inv = self.divider_quantizer(tf.cast(1., tf.float32) / self.n)
        else:
            n_inv = 1. / self.n

        sum = tf.math.reduce_sum(input, axis=-1)
        output = tf.math.multiply(sum, n_inv)

        if self.kernel_quantizer is not None:
            quantized_output = self.kernel_quantizer(output)
        else:
            quantized_output = output

        return quantized_output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        return super(QGlobalAverageReduce1D, self).get_config()

    def get_quantization_config(self):
        return {
            "kernel_quantizer": str(self.kernel_quantizer),
            "divider_quantizer": str(self.kernel_quantizer),
            "units": str(self.units),
        }

    def get_quantizers(self):
        return [self.kernel_quantizer, self.divider_quantizer]
