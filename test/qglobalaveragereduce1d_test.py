import pytest
import logging

import tensorflow as tf
import numpy as np

from qgravnet import QGlobalAverageReduce1D

LOGGER = logging.getLogger(__name__)

@pytest.mark.parametrize(
    'kernel_quantizer, divider_quantizer, inputs, expected_output',
    [
        (
            None, None,
            tf.constant([[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.]]),
            tf.constant([4.5])
        ),
        (
            "quantized_bits(4,4,1)", None,
            tf.constant([[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.]]),
            tf.constant([4.])
        ),
        (
           None, None,
            tf.constant([[[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.],
                          [0.,1.,2.,3.,4.,5.,6.,7.,8.,9.]]]),
            tf.constant([[4.5,4.5]])
        ),
        (
           "quantized_bits(4,4,1)", "quantized_bits(3,4,1)",
           tf.constant([[[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.],
                        [0.,1.,2.,3.,4.,5.,6.,7.,8.,9.]]]),
           tf.constant([[6.,6.]])
        ),
    ])
def test_qlobalaveragereduce1d(kernel_quantizer, divider_quantizer, inputs, expected_output):
    outputs = QGlobalAverageReduce1D(kernel_quantizer, divider_quantizer)(inputs)
    np.testing.assert_array_equal(outputs.numpy(),
                                  expected_output.numpy())

if __name__ == '__main__':
  pytest.main([__file__])