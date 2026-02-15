import pytest
import logging

import tensorflow as tf
import numpy as np

from qgravnet import QExponential

LOGGER = logging.getLogger(__name__)

@pytest.mark.parametrize(
    'alpha, exponential_quantizer ,inputs, expected_outputs',
    [
        (
        	1.0, None,
			tf.constant([[[0.,0.],
                          [1.,1.],
                          [2.,2.]]]),
            tf.constant([[[1.,1.],
                          [2.718281,2.718281],
                          [7.38905159496,7.38905159496]]])
        ),
        (
        	-1.0, None,
			tf.constant([[[0.,0.],
                          [1.,1.],
                          [2.,2.]]]),
            tf.constant([[[1.,1.],
                          [0.36787955329 ,0.36787955329 ],
                          [0.13533536573 ,0.13533536573 ]]])
        ),
        (
        	-1.0, "quantized_exp(8,0,1)",
			tf.constant([[[0.,0.],
                          [1.,1.],
                          [2.,2.]]]),
            tf.constant([[[1.,1.],
                          [0.36787955329 ,0.36787955329 ],
                          [0.13533536573 ,0.13533536573 ]]])
        ),
    ])

def test_qexponential(alpha, exponential_quantizer, inputs, expected_outputs):
    outputs = QExponential(alpha,exponential_quantizer)(inputs)
    for output,expected_output in zip(outputs,expected_outputs):
        np.testing.assert_allclose(output.numpy(), expected_output.numpy(),rtol=0.02,atol=0.001)

if __name__ == '__main__':
  pytest.main([__file__])
