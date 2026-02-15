import pytest
import logging

import tensorflow as tf
import numpy as np

from qgravnet import QGlobalExchange

LOGGER = logging.getLogger(__name__)

@pytest.mark.parametrize(
    'kernel_quantizer, divider_quantizer, inputs, expected_output',
    [
        (
            None, None,
            [tf.constant([[[0.,1.,2.,3.],
                           [4.,5.,6.,7.],
                           [8.,9.,10.,11.]]]),
             tf.constant([[3.]])],
            tf.constant([[[0.,1.,2.,3.,4.,5.,6.,7.],
                          [4.,5.,6.,7.,4.,5.,6.,7.],
                          [8.,9.,10.,11.,4.,5.,6.,7.]]])
        ),
        (
            "quantized_bits(4,3,1)", "quantized_bits(2,2,1)",
            [tf.constant([[[0.,1.,2.,3.],
                           [4.,5.,6.,7.],
                           [8.,9.,10.,11.]]]),
             tf.constant([[2.]])],
            tf.constant([[[0.,1.,2.,3.,2.,3.,4.,5.],
                          [4.,5.,6.,7.,2.,3.,4.,5.],
                          [0.,0.,0.,0.,2.,3.,4.,5.]]])
        ),
        (
            None, None,
            [tf.constant([[[0.,1.,2.,3.],
                           [4.,5.,6.,7.],
                           [8.,9.,10.,11.]]]),
             tf.constant([[2.]])],
            tf.constant([[[0.,1.,2.,3.,2.,3.,4.,5.],
                          [4.,5.,6.,7.,2.,3.,4.,5.],
                          [0.,0.,0.,0.,2.,3.,4.,5.]]])
        ),
    ])
def test_qglobalexchange(kernel_quantizer, divider_quantizer, inputs, expected_output):
    output = QGlobalExchange(kernel_quantizer, divider_quantizer)(inputs)
    np.testing.assert_array_equal(output.numpy(),
                                  expected_output.numpy())

if __name__ == '__main__':
  pytest.main([__file__])