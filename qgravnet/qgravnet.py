import tensorflow.keras as keras
import tensorflow as tf

K = keras.backend

from qkeras.quantizers import get_quantizer
from qkeras import QDense, QActivation


class NamedQDense(QDense):
    def add_weight(self, name=None, **kwargs):
        return super(NamedQDense, self).add_weight(
            name="%s_%s" % (self.name, name), **kwargs
        )


class NamedQActivation(QActivation):
    def add_weight(self, name=None, **kwargs):
        return super(NamedQActivation, self).add_weight(
            name="%s_%s" % (self.name, name), **kwargs
        )


def euclidean_squared(A, B):
    """
    Returns euclidean distance between two batches of shape [B,N,F] and [B,M,F] where B is batch size, N is number of
    examples in the batch of first set, M is number of examples in the batch of second set, F is number of spatial
    features.

    Returns:
    A matrix of size [B, N, M] where each element [i,j] denotes euclidean distance between ith entry in first set and
    jth in second set.

    """

    shape_A = A.get_shape().as_list()
    shape_B = B.get_shape().as_list()

    assert len(shape_A) == 3 and len(shape_B) == 3
    assert shape_A[0] == shape_B[0]  # and shape_A[1] == shape_B[1]

    sub_factor = -2 * tf.matmul(A, tf.transpose(B, perm=[0, 2, 1]))  # -2ab term
    dotA = tf.expand_dims(tf.reduce_sum(A * A, axis=2), axis=2)  # a^2 term
    dotB = tf.expand_dims(tf.reduce_sum(B * B, axis=2), axis=1)  # b^2 term
    return tf.abs(sub_factor + dotA + dotB)


def manhattan(A, B):
    """
    Returns manhattan distance between two batches of shape [B,N,F] and [B,M,F] where B is batch size, N is number of
    examples in the batch of first set, M is number of examples in the batch of second set, F is number of spatial
    features.

    Returns:
    A matrix of size [B, N, M] where each element [i,j] denotes manhattan distance between ith entry in first set and
    jth in second set.

    """

    shape_A = A.get_shape().as_list()
    shape_B = B.get_shape().as_list()

    assert len(shape_A) == 3 and len(shape_B) == 3
    assert shape_A[0] == shape_B[0]  # and shape_A[1] == shape_B[1]

    return tf.transpose(
        tf.reduce_sum(tf.abs(A - tf.expand_dims(B, axis=1)), axis=-1), perm=[0, 2, 1]
    )


def nearest_neighbor_matrix(spatial_features, k=10):
    """
    Nearest neighbors matrix given spatial features.

    :param spatial_features: Spatial features of shape [B, N, S] where B = batch size, N = max examples in batch,
                             S = spatial features
    :param k: Max neighbors
    :return:
    """

    shape = spatial_features.get_shape().as_list()

    assert len(shape) == 3

    D = euclidean_squared(spatial_features, spatial_features)
    D, N = tf.nn.top_k(-D, k)
    return N, -D


# Hack keras Dense to propagate the layer name into saved weights
class NamedDense(keras.layers.Dense):
    def add_weight(self, name=None, **kwargs):
        return super(NamedDense, self).add_weight(
            name="%s_%s" % (self.name, name), **kwargs
        )


# Hack keras Activation to propagate the layer name into saved weights
class NamedActivation(keras.layers.Activation):
    def add_weight(self, name=None, **kwargs):
        return super(NamedActivation, self).add_weight(
            name="%s_%s" % (self.name, name), **kwargs
        )


class QGravNet(keras.layers.Layer):
    def __init__(
        self,
        n_neighbours,
        n_dimensions,
        n_filters,
        n_propagate,
        layer_name,
        also_coordinates=False,
        feature_dropout=-1,
        coordinate_feature_quantizer=None,
        spatial_feature_quantizer=None,
        distance_quantizer=None,
        exponential_quantizer=None,
        output_feature_quantizer=None,
        other_kernel_initializer="glorot_uniform",
        coordinate_kernel_initializer=keras.initializers.Orthogonal(),
        fix_coordinate_space=False,
        masked_coordinate_offset=None,
        output_activation=None,
        quantize_transforms=False,
        use_manhattan_distance=False,
        exponential_factor=10.0,
        **kwargs
    ):
        super(QGravNet, self).__init__(**kwargs)

        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        self.n_filters = n_filters
        self.n_propagate = n_propagate
        self.layer_name = layer_name
        self.also_coordinates = also_coordinates
        self.feature_dropout = feature_dropout
        self.masked_coordinate_offset = masked_coordinate_offset
        self.quantize_transforms = quantize_transforms
        self.use_manhattan_distance = use_manhattan_distance
        self.exponential_factor = exponential_factor

        if self.quantize_transforms:
            self.input_feature_transform = NamedQDense(
                n_propagate,
                name=layer_name + "_FLR",
                kernel_initializer=other_kernel_initializer,
                kernel_quantizer=get_quantizer(spatial_feature_quantizer),
                bias_quantizer=get_quantizer(spatial_feature_quantizer),
            )
            self.input_spatial_transform = NamedQDense(
                n_dimensions,
                name=layer_name + "_S",
                kernel_initializer=coordinate_kernel_initializer,
                kernel_quantizer=get_quantizer(coordinate_feature_quantizer),
                bias_quantizer=get_quantizer(coordinate_feature_quantizer),
            )
            self.output_feature_transform = NamedQDense(
                n_filters,
                name=layer_name + "_Fout",
                kernel_initializer=other_kernel_initializer,
                kernel_quantizer=get_quantizer(output_feature_quantizer),
                bias_quantizer=get_quantizer(output_feature_quantizer),
                activation=output_activation,
            )
            self.distance_quantization = get_quantizer(distance_quantizer)
            self.exponential_quantization = get_quantizer(exponential_quantizer)
        else:
            self.input_feature_transform = NamedDense(
                n_propagate,
                name=layer_name + "_FLR",
                kernel_initializer=other_kernel_initializer,
            )
            self.input_spatial_transform = NamedDense(
                n_dimensions,
                name=layer_name + "_S",
                kernel_initializer=coordinate_kernel_initializer,
            )
            self.output_feature_transform = NamedDense(
                n_filters,
                activation=output_activation,
                name=layer_name + "_Fout",
                kernel_initializer=other_kernel_initializer,
            )

        self._sublayers = [
            self.input_feature_transform,
            self.input_spatial_transform,
            self.output_feature_transform,
        ]
        if fix_coordinate_space:
            self.input_spatial_transform = None
            self._sublayers = [
                self.input_feature_transform,
                self.output_feature_transform,
            ]

    def build(self, input_shape):
        if self.masked_coordinate_offset is not None:
            input_shape = input_shape[0]

        self.input_feature_transform.build(input_shape)
        if self.input_spatial_transform is not None:
            self.input_spatial_transform.build(input_shape)

        self.output_feature_transform.build(
            (
                input_shape[0],
                input_shape[1],
                input_shape[2] + self.input_feature_transform.units * 2,
            )
        )

        for layer in self._sublayers:
            self._trainable_weights.extend(layer.trainable_weights)
            self._non_trainable_weights.extend(layer.non_trainable_weights)

        super(QGravNet, self).build(input_shape)

    def call(self, x):

        if self.masked_coordinate_offset is not None:
            if not isinstance(x, list):
                raise Exception(
                    "GravNet: in mask mode, input must be list of input,mask"
                )
            mask = x[1]
            x = x[0]

        features = self.input_feature_transform(x)

        if self.feature_dropout > 0 and self.feature_dropout < 1:
            features = keras.layers.Dropout(self.feature_dropout)(features)

        if self.input_spatial_transform is not None:
            coordinates = self.input_spatial_transform(x)
        else:
            coordinates = x[:, :, 0 : self.n_dimensions]

        if self.masked_coordinate_offset is not None:
            sel_mask = K.tile(mask, [1, 1, K.shape(coordinates)[2]])
            coordinates = K.switch(
                K.greater(sel_mask, 0.0),
                coordinates,
                K.zeros_like(coordinates) - self.masked_coordinate_offset,
            )

        collected_neighbours = self.collect_neighbours(coordinates, features)

        updated_features = K.concatenate([x, collected_neighbours], axis=-1)
        output = self.output_feature_transform(updated_features)

        if self.masked_coordinate_offset is not None:
            output *= mask

        if self.also_coordinates:
            return [output, coordinates]
        return output

    def compute_output_shape(self, input_shape):
        if self.masked_coordinate_offset is not None:
            input_shape = input_shape[0]
        if self.also_coordinates:
            return [
                (input_shape[0], input_shape[1], self.output_feature_transform.units),
                (input_shape[0], input_shape[1], self.n_dimensions),
            ]

        # tf.ragged FIXME? tf.shape() might do the trick already
        return (input_shape[0], input_shape[1], self.output_feature_transform.units)

    def collect_neighbours(self, coordinates, features):
        # tf.ragged FIXME?
        # for euclidean_squared see caloGraphNN.py
        if self.use_manhattan_distance:
            distance_matrix = manhattan(coordinates, coordinates)
        else:
            distance_matrix = euclidean_squared(coordinates, coordinates)

        ranked_distances, ranked_indices = tf.nn.top_k(
            -distance_matrix, self.n_neighbours
        )

        neighbour_indices = ranked_indices[:, :, 1:]

        n_batches = tf.shape(features)[0]

        # tf.ragged FIXME? or could that work?
        n_vertices = K.shape(features)[1]
        n_features = K.shape(features)[2]

        batch_range = K.arange(n_batches)
        batch_range = K.expand_dims(batch_range, axis=1)
        batch_range = K.expand_dims(batch_range, axis=1)
        batch_range = K.expand_dims(batch_range, axis=1)  # (B, 1, 1, 1)

        # tf.ragged FIXME? n_vertices
        batch_indices = K.tile(
            batch_range, [1, n_vertices, self.n_neighbours - 1, 1]
        )  # (B, V, N-1, 1)
        vertex_indices = K.expand_dims(neighbour_indices, axis=3)  # (B, V, N-1, 1)
        indices = K.concatenate([batch_indices, vertex_indices], axis=-1)

        neighbour_features = tf.gather_nd(features, indices)  # (B, V, N-1, F)

        distance = -ranked_distances[:, :, 1:]

        # Quantize Distances
        if self.quantize_transforms:
            # This quantization is essentially part of the calculate distance kernel.
            distance_quantized = self.distance_quantization(
                distance * self.exponential_factor
            )
            weights_quantized = self.exponential_quantization(distance_quantized)
            weights = K.expand_dims(weights_quantized, axis=-1)
        else:
            weights = tf.exp(-1 * (tf.abs(distance * self.exponential_factor)))
            weights = K.expand_dims(weights, axis=-1)

        # weight the neighbour_features
        neighbour_features *= weights

        neighbours_max = K.max(neighbour_features, axis=2)
        neighbours_mean = K.mean(neighbour_features, axis=2)

        return K.concatenate([neighbours_max, neighbours_mean], axis=-1)

    def get_config(self):
        config = {
            "n_neighbours": self.n_neighbours,
            "n_dimensions": self.n_dimensions,
            "n_filters": self.n_filters,
            "n_propagate": self.n_propagate,
            "name": self.name,
            "also_coordinates": self.also_coordinates,
            "feature_dropout": self.feature_dropout,
            "masked_coordinate_offset": self.masked_coordinate_offset,
        }
        base_config = super(QGravNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
