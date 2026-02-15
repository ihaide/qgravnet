import pytest
import logging

import tensorflow as tf
import numpy as np

from qgravnet import QNearestNeighbour

LOGGER = logging.getLogger(__name__)

@pytest.mark.parametrize(
    'k, distance_quantizer ,inputs, expected_outputs',
    [
        (
        	3, None,
			[tf.constant([[[0.,0.],
                           [1.,1.],
                           [2.,2.],
                           [3.,3.],
                           [4.,4.]]]),
    		 tf.constant([[[0.,1.,2.],
                     	   [3.,4.,5.],
                           [6.,7.,8.],
                           [9.,10.,11.],
                           [12.,13.,14.]]]),
             tf.constant([[5.]])],
            [tf.constant([[[2.,8.],
                           [2.,2.],
                     	   [2.,2.],
                		   [2.,2.],
                           [2.,8.]]]),
             tf.constant([[[[3.,4,5.],[6.,7.,8.]],
                           [[0.,1.,2],[6.,7.,8.]],
                     	   [[3.,4.,5],[9.,10.,11.]],
                     	   [[6.,7.,8],[12.,13.,14.]],
                           [[9.,10.,11],[6.,7.,8.]]]])]
        ),
        (
        	2, None,
			[tf.constant([[[0.,0.],
                           [1.,1.],
                           [2.,2.],
                           [3.,3.],
                           [4.,4.]]]),
    		 tf.constant([[[0.,1.,2.],
                     	   [3.,4.,5.],
                           [6.,7.,8.],
                           [9.,10.,11.],
                           [12.,13.,14.]]]),
             tf.constant([[5.]])],
            [tf.constant([[[2.],
                           [2.],
                     	   [2.],
                		   [2.],
                           [2.]]]),
             tf.constant([[[[3.,4.,5.]],
                           [[0.,1.,2.]],
                     	   [[3.,4.,5.]],
                     	   [[6.,7.,8.]],
                           [[9.,10.,11.]]]])]
        ),
        (
        	2, "quantized_bits(1,1,1)",
			[tf.constant([[[0.,0.],
                           [1.,1.],
                           [2.,2.],
                           [3.,3.],
                           [4.,4.]]]),
    		 tf.constant([[[0.,1.,2.],
                     	   [3.,4.,5.],
                           [6.,7.,8.],
                           [9.,10.,11.],
                           [12.,13.,14.]]]),
             tf.constant([[5.]])],
            [tf.constant([[[1.],
                           [1.],
                     	   [1.],
                		   [1.],
                           [1.]]]),
             tf.constant([[[[3.,4.,5.]],
                           [[0.,1.,2.]],
                     	   [[3.,4.,5.]],
                     	   [[6.,7.,8.]],
                           [[9.,10.,11.]]]])]
        ),
    ])
def test_qnearestneighbour(k, distance_quantizer, inputs, expected_outputs):
    outputs = QNearestNeighbour(k,distance_quantizer)(inputs)
    for output,expected_output in zip(outputs,expected_outputs):
        np.testing.assert_array_equal(output.numpy(), expected_output.numpy())

@pytest.mark.parametrize(
    'k, distance_quantizer ,inputs, expected_outputs',
    [
        (
        	2, None,
			[tf.constant([[[0.,0.],
                           [1.,1.],
                           [2.,2.],
                           [3.,3.],
                           [4.,4.]],
                           [[0.,0.],
                           [1.,1.],
                           [2.,2.],
                           [0.,0.],
                           [0.,0.]]]),
    		 tf.constant([[[0.,1.,2.],
                     	   [3.,4.,5.],
                           [6.,7.,8.],
                           [9.,10.,11.],
                           [12.,13.,14.]],
                           [[0.,1.,2.],
                     	   [3.,4.,5.],
                           [6.,7.,8.],
                           [9.,10.,11.],
                           [12.,13.,14.]]]),
             tf.constant([[5.],[3.]])],
            [tf.constant([[[2.],
                           [2.],
                     	   [2.],
                		   [2.],
                           [2.]],
                           [[2.],
                           [2.],
                     	   [2.],
                		   [np.finfo(np.float32).max],
                           [np.finfo(np.float32).max]]]),
             tf.constant([[[[3.,4.,5.]],
                          [[0.,1.,2.]],
                     	  [[3.,4.,5.]],
                     	  [[6.,7.,8.]],
                          [[9.,10.,11.]]],
                          [[[3.,4.,5.]],
                          [[0.,1.,2.]],
                     	  [[3.,4.,5.]],
                     	  [[0.,0.,0.]],
                          [[0.,0.,0.]]]])]
        )
    ])
def test_mask_qnearestneighbour(k, distance_quantizer, inputs, expected_outputs):
    outputs = QNearestNeighbour(k,distance_quantizer)(inputs)
    #Test Distances
    np.testing.assert_array_equal(outputs[0].numpy(), expected_outputs[0].numpy())
    #Test Features. We do not care about masked features
    for expected_feature, \
        output_feature, \
        n_vertices in zip(expected_outputs[1].numpy(), \
                          outputs[1].numpy(), \
                          inputs[2].numpy()):
        #Select only active feature outputs
        i = int(n_vertices[0])
        expected_feature = expected_feature[:i,:]
        output_feature = output_feature[:i,:]
        np.testing.assert_array_equal(output_feature, expected_feature)

if __name__ == '__main__':
  pytest.main([__file__])
