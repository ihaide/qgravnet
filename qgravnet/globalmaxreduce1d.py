import tensorflow.keras as keras
import tensorflow as tf


class GlobalMaxReduce1D(keras.layers.Layer):
    """
    This Layer Reduces the innermost dimension of an arbitrary input tensor
    by calculating the Max() funcion on it.
    Args:
     - None
    Shapes:
     - **:input:**
       Tensor of rank
     - **:output:**
       Input tensor rank-1
    """

    def __init__(self, **kwargs):
        super(GlobalMaxReduce1D, self).__init__(**kwargs)

    def call(self, inputs):
        return keras.backend.max(inputs, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        return super(GlobalMaxReduce1D, self).get_config()
