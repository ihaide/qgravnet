import pytest
import logging

import tensorflow as tf
import numpy as np

from qgravnet import GlobalMaxReduce1D

LOGGER = logging.getLogger(__name__)

@pytest.mark.parametrize(
    'inputs, expected_output',
    [
        (
            tf.constant([[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.]]),
            tf.constant([9.])
        ),
        (
            tf.constant([[[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.],
                          [0.,1.,2.,3.,4.,5.,6.,7.,8.,9.]]]),
            tf.constant([[9.,9.]])
        ),
    ])
def test_globalmaxreduce1d(inputs, expected_output):
    outputs = GlobalMaxReduce1D()(inputs)
    np.testing.assert_array_equal(outputs.numpy(),
                                  expected_output.numpy())

if __name__ == '__main__':
  pytest.main([__file__])