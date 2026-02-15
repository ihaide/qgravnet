import sys

import numpy as np

import tensorflow.keras as keras
import tensorflow as tf

K = keras.backend

from qkeras.quantizers import get_quantizer


class QNearestNeighbourEuclidean(keras.layers.Layer):
    """
    Quantized implementation of the k nearest neighbour algorihtm. Euclediean or manhattan
    distances is used for calculating the metric. Quantization is additionally
    activated by setting  distance_quantizer and/or distance_activation.

    Args:
        - k: Number of neighbours to select
        - distance_quantizer: QKeras quantizer indicating the precision of the
          calculated distances.

    Shapes:
        - **input:**
          List of tensors [coordinates, features, n_vertices]
          coordinates :math:`(|\\mathcal{V}|, |\\mathcal{S}|)`
          features :math:`(|\\mathcal{V}|, |\\mathcal{F}|)`
          n_vertices :math:`(1)`
        - **output:**
          List of tensors [distances, neighbour_features]:
          distances :math:`(|\\mathcal{V}|, |\\mathcal{K-1}|)`
          neighbour_features :math:`(|\\mathcal{V}|, |\\mathcal{K-1}|, |\\mathcal{F}|)`
     -
    """

    def __init__(
        self,
        k,
        distance_quantizer=None,
        use_manhattan_distance=False,
        max_distance=None,
        **kwargs,
    ):
        self.k = k
        self.distance_quantizer = (
            get_quantizer(distance_quantizer)
            if distance_quantizer is not None
            else None
        )
        if max_distance is None:
            self.max_distance = (
                np.finfo(np.float32).max
                if self.distance_quantizer is None
                else 2 ** int(distance_quantizer.split("(")[1].split(",")[0])
            )
        else:
            self.max_distance = max_distance
        self.use_manhattan_distance = use_manhattan_distance
        super(QNearestNeighbourEuclidean, self).__init__(**kwargs)

    def build(self, input_shapes):
        """
        Precalculate the raw mask. Is later used determine individual event size

        Raw Mask:
        0 1 2 . V
        1 2 3 . V
        2 3 4 . V
        . . . . V
        V V V V V

        """
        v = input_shapes[0][1]

        rows = tf.tile(tf.expand_dims(tf.range(v), axis=0), [v, 1])
        columns = tf.tile(tf.expand_dims(tf.range(v), axis=-1), [1, v])
        raw_mask = tf.maximum(rows, columns)
        self.raw_mask = tf.cast(raw_mask, dtype=tf.float32)

        super(QNearestNeighbourEuclidean, self).build(input_shapes)

    def call(self, inputs):
        coordinates = inputs[0]
        features = inputs[1]
        active_vertices = inputs[2]

        # Calculate squared euclidean distances as (a-b)^2 = a^2 + b^2 - 2ab
        sub_factor = -2 * tf.matmul(
            coordinates, tf.transpose(coordinates, perm=[0, 2, 1])
        )  # -2ab term
        dotA = tf.expand_dims(
            tf.reduce_sum(coordinates * coordinates, axis=2), axis=2
        )  # a^2 term
        dotB = tf.expand_dims(
            tf.reduce_sum(coordinates * coordinates, axis=2), axis=1
        )  # b^2 term
        distance_matrix = tf.abs(sub_factor + dotA + dotB)  # (B,V,V)

        b = tf.shape(inputs[0])[0]
        # 1. Expand static raw mask to batch size from (V,V) to (B,V,V)
        batch_raw_mask = tf.tile(tf.expand_dims(self.raw_mask, axis=0), [b, 1, 1])
        # 2. Expand actvie vertices tensor from (B,1) to (B,1,1)
        batch_active_vertices = tf.expand_dims(active_vertices, axis=-1)
        # 3. Calculate boolean mask
        mask = tf.less(batch_raw_mask, batch_active_vertices)
        # 4. Apply mask. Set all invalid distances to the largest float value on the current system.
        distance_matrix = tf.where(
            mask, distance_matrix, tf.zeros_like(distance_matrix) + self.max_distance
        )

        # Sort distances and select k smallest
        ranked_distances, ranked_indices = tf.nn.top_k(
            -distance_matrix, self.k
        )  # (B,V,K)

        # Remove self loop. The first distance value must always be equal to 0.
        ranked_indices = ranked_indices[:, :, 1:]  # (B,V,K-1)
        neighbour_distances = -ranked_distances[:, :, 1:]  # (B,V,K-1)

        # Gather all neighbours. We have to expand the index tensor, because we want to retrieve vertices features in the last dimension F
        # By batch_dims=1 we implicity perform the gather step for every batch seperately.
        neighbour_indices = tf.expand_dims(ranked_indices, axis=-1)  # (B,V,K-1,1)
        neighbour_features = tf.gather_nd(
            batch_dims=1, params=features, indices=neighbour_indices
        )  # (B,V,K-1,F)

        if self.distance_quantizer is not None:
            quantized_distances = self.distance_quantizer(neighbour_distances)
        else:
            quantized_distances = neighbour_distances

        return [quantized_distances, neighbour_features]

    def compute_output_shape(self, input_shapes):

        coordinate_shape = input_shapes[0]
        feature_shape = input_shapes[1]
        active_vertices_shape = input_shapes[2]

        assert len(coordinate_shape) == 3
        assert len(feature_shape) == 3
        assert len(active_vertices_shape) == 2

        assert coordinate_shape[0] == feature_shape[0]

        v = coordinate_shape[0]
        f = feature_shape[1]

        distance_shape = (v, self.k - 1)
        neighbour_feature_shape = (v, self.k - 1, f)

        return [distance_shape, neighbour_feature_shape]

    def get_config(self):
        config = {
            "k": self.k,
        }
        base_config = super(QNearestNeighbourEuclidean, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_quantization_config(self):
        return {
            "distance_quantizer": str(self.distance_quantizer),
            "units": str(self.units),
        }

    def get_quantizers(self):
        return [self.distance_quantizer]


class QNearestNeighbourManhattan(keras.layers.Layer):
    """
    Quantized implementation of the k nearest neighbour algorihtm. Euclediean or manhattan
    distances is used for calculating the metric. Quantization is additionally
    activated by setting  distance_quantizer and/or distance_activation.

    Args:
        - k: Number of neighbours to select
        - distance_quantizer: QKeras quantizer indicating the precision of the
          calculated distances.

    Shapes:
        - **input:**
          List of tensors [coordinates, features, n_vertices]
          coordinates :math:`(|\\mathcal{V}|, |\\mathcal{S}|)`
          features :math:`(|\\mathcal{V}|, |\\mathcal{F}|)`
          n_vertices :math:`(1)`
        - **output:**
          List of tensors [distances, neighbour_features]:
          distances :math:`(|\\mathcal{V}|, |\\mathcal{K-1}|)`
          neighbour_features :math:`(|\\mathcal{V}|, |\\mathcal{K-1}|, |\\mathcal{F}|)`
     -
    """

    def __init__(
        self,
        k,
        distance_quantizer=None,
        use_manhattan_distance=False,
        max_distance=None,
        **kwargs,
    ):
        self.k = k
        self.distance_quantizer = (
            get_quantizer(distance_quantizer)
            if distance_quantizer is not None
            else None
        )
        if max_distance is None:
            self.max_distance = (
                np.finfo(np.float32).max
                if self.distance_quantizer is None
                else 2 ** int(distance_quantizer.split("(")[1].split(",")[0])
            )
        else:
            self.max_distance = max_distance
        self.use_manhattan_distance = use_manhattan_distance
        super(QNearestNeighbourManhattan, self).__init__(**kwargs)

    def build(self, input_shapes):
        """
        Precalculate the raw mask. Is later used determine individual event size

        Raw Mask:
        0 1 2 . V
        1 2 3 . V
        2 3 4 . V
        . . . . V
        V V V V V

        """
        v = input_shapes[0][1]

        rows = tf.tile(tf.expand_dims(tf.range(v), axis=0), [v, 1])
        columns = tf.tile(tf.expand_dims(tf.range(v), axis=-1), [1, v])
        raw_mask = tf.maximum(rows, columns)
        self.raw_mask = tf.cast(raw_mask, dtype=tf.float32)

        super(QNearestNeighbourManhattan, self).build(input_shapes)

    def call(self, inputs):
        coordinates = inputs[0]
        features = inputs[1]
        active_vertices = inputs[2]

        # Calculate manhattan distances as |a-b|
        distance_matrix = tf.reduce_sum(
            tf.abs(
                tf.expand_dims(coordinates, axis=2)
                - tf.expand_dims(coordinates, axis=1)
            ),
            axis=-1,
        )
        # (B,V,V)

        b = tf.shape(inputs[0])[0]
        # 1. Expand static raw mask to batch size from (V,V) to (B,V,V)
        batch_raw_mask = tf.tile(tf.expand_dims(self.raw_mask, axis=0), [b, 1, 1])
        # 2. Expand actvie vertices tensor from (B,1) to (B,1,1)
        batch_active_vertices = tf.expand_dims(active_vertices, axis=-1)
        # 3. Calculate boolean mask
        mask = tf.less(batch_raw_mask, batch_active_vertices)
        # 4. Apply mask. Set all invalid distances to the largest float value on the current system.
        distance_matrix = tf.where(
            mask, distance_matrix, tf.zeros_like(distance_matrix) + self.max_distance
        )

        # Sort distances and select k smallest
        ranked_distances, ranked_indices = tf.nn.top_k(
            -distance_matrix, self.k
        )  # (B,V,K)

        # Remove self loop. The first distance value must always be equal to 0.
        ranked_indices = ranked_indices[:, :, 1:]  # (B,V,K-1)
        neighbour_distances = -ranked_distances[:, :, 1:]  # (B,V,K-1)

        # Gather all neighbours. We have to expand the index tensor, because we want to retrieve vertices features in the last dimension F
        # By batch_dims=1 we implicity perform the gather step for every batch separately.
        neighbour_indices = tf.expand_dims(ranked_indices, axis=-1)  # (B,V,K-1,1)
        neighbour_features = tf.gather_nd(
            batch_dims=1, params=features, indices=neighbour_indices
        )  # (B,V,K-1,F)

        if self.distance_quantizer is not None:
            quantized_distances = self.distance_quantizer(neighbour_distances)
        else:
            quantized_distances = neighbour_distances

        return [quantized_distances, neighbour_features]

    def compute_output_shape(self, input_shapes):

        coordinate_shape = input_shapes[0]
        feature_shape = input_shapes[1]
        active_vertices_shape = input_shapes[2]

        assert len(coordinate_shape) == 3
        assert len(feature_shape) == 3
        assert len(active_vertices_shape) == 2

        assert coordinate_shape[0] == feature_shape[0]

        v = coordinate_shape[0]
        f = feature_shape[1]

        distance_shape = (v, self.k - 1)
        neighbour_feature_shape = (v, self.k - 1, f)

        return [distance_shape, neighbour_feature_shape]

    def get_config(self):
        config = {
            "k": self.k,
        }
        base_config = super(QNearestNeighbourManhattan, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_quantization_config(self):
        return {
            "distance_quantizer": str(self.distance_quantizer),
            "units": str(self.units),
        }

    def get_quantizers(self):
        return [self.distance_quantizer]
