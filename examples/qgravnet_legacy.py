import sys
import datetime, os
import json

import tensorflow as tf
from tensorflow.keras.layers import Multiply, GlobalMaxPooling1D, GlobalAveragePooling1D,Concatenate
from tensorflow.keras.backend import expand_dims

from qgravnet import QGravNet, QNearestNeighbour,QExponential,GlobalMaxReduce1D,QGlobalAverageReduce1D
from qkeras import QDense, QActivation

class SimpleModel(tf.keras.Model):
    def __init__(self, k=16, s=4, f_lr=12, f_out=18, **kwargs):
        super(SimpleModel, self).__init__(**kwargs)
        self.gravnet = QGravNet(k, s, f_lr, f_out, 'test',
                 coordinate_feature_quantizer="quantized_bits(8,0,1)",
                 spatial_feature_quantizer="quantized_bits(8,0,1)",
                 distance_quantizer="quantized_bits(8,0,1)",
                 exponential_quantizer="quantized_exp(8,0,1)",
                 output_feature_quantizer="quantized_bits(8,0,1)",
                 quantize_transforms=True)

    def call(self, inputs):
        return self.gravnet(inputs)
    
model = SimpleModel()
x = tf.random.normal(shape=(10, 128, 8))# BxNxF
y = model(x) 

model.summary()