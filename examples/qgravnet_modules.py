#%%
import sys
import datetime, os
import json

import tensorflow as tf
from tensorflow.keras.layers import Multiply, Concatenate, Permute
from tensorflow.keras.backend import expand_dims

from qgravnet import QGravNet, QNearestNeighbour,QExponential,GlobalMaxReduce1D,QGlobalAverageReduce1D
from qkeras import QDense, QActivation


vertices = tf.keras.Input(shape=(64, 16), name="vertices")
n_vertices = tf.keras.Input(shape=(1), name="n_vertices")

coordinate_transform = QDense(4)(vertices)
feature_transform = QDense(12)(vertices)
neighbour_distances, neighbour_features = QNearestNeighbour(k=4)([coordinate_transform,feature_transform, n_vertices])
weighted_distances = QExponential(alpha=-1.0)(neighbour_distances)
weighted_features = Multiply()([expand_dims(weighted_distances, axis=-1), neighbour_features])
permuted_features = Permute(dims=(1,3,2))(weighted_features)
average_features = QGlobalAverageReduce1D()(permuted_features)
max_features = GlobalMaxReduce1D()(permuted_features)
all_features = Concatenate()([vertices, average_features, max_features])
output_transform = QDense(10)(all_features)

model = tf.keras.Model(inputs=[vertices,n_vertices], outputs=[output_transform], name="QGravNet")


model.summary()


model_json = model.to_json()
pretty_model_json = json.loads(model_json)
print(json.dumps(pretty_model_json, indent=4))
tf.keras.utils.plot_model(model, "test.png", show_shapes=True)
# %%
