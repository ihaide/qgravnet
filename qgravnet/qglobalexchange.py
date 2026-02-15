import tensorflow.keras as keras
import tensorflow as tf

from qkeras.quantizers import get_quantizer

class QGlobalExchange(keras.layers.Layer):
    """
    Global Exchange Layer. Appends Mean of Features to each vertex
    Args:
     - kernel_quantizer: Quantization of the output value. Should be of type
       quantized_exp(...)
     - divider_quantizer: Quantization of the divider value. Should be of type
       quantized_exp(...)
    Shapes:
     - **:inputs:**
       List of tensors [features, active_vertices]
       features :math:`(|\\mathcal{V}|, |\\mathcal{F}|)`
       n_vertices :math:`(1)` 
     - **:output:**
       Tensor :math:`(|\\mathcal{V}|, |\\mathcal{2F}|)`

    """
    def __init__(self,kernel_quantizer=None, divider_quantizer=None):
        if kernel_quantizer is not None:
            self.kernel_quantizer = get_quantizer(kernel_quantizer)
        else:
            self.kernel_quantizer = None
        if divider_quantizer is not None:
            self.divider_quantizer = get_quantizer(divider_quantizer)
        else:
            self.divider_quantizer = None
        super(QGlobalExchange, self).__init__(trainable=False)

    def build(self, input_shapes):
        # We expect two tensors
        assert len(input_shapes) == 2
        # First Tensor (B,V,F) 
        assert input_shapes[0][0] == input_shapes[1][0]
        # Second Tensor (B,1)
        assert input_shapes[1][0] == 1

        self.v = input_shapes[0][1]
        self.f = input_shapes[0][2]

        raw_mask = tf.tile(tf.expand_dims(tf.range(self.v),axis=-1), [1,self.f])
        self.raw_mask = tf.cast(raw_mask,dtype=tf.float32)

        super(QGlobalExchange, self).build(input_shapes)

    def call(self, inputs):
        features = inputs[0] # (B,V,F)
        active_vertices = inputs[1] # (B,1)

        if self.divider_quantizer is not None:
            quantized_active_vertices = self.divider_quantizer(active_vertices)
        else:
            quantized_active_vertices = active_vertices

        b = tf.shape(inputs[0])[0]
        #Apply mask on input vertices
        # 1. Expand static raw mask to batch size from (V,F) to (B,V,F)
        batch_raw_mask = tf.tile(tf.expand_dims(self.raw_mask,axis=0), [b, 1, 1])
        # 2. Expand actvie vertices tensor from (B,1) to (B,1,1)
        batch_active_vertices = tf.expand_dims(quantized_active_vertices,axis=-1)
        # 3. Calculate boolean mask
        mask = tf.less(batch_raw_mask,batch_active_vertices)
        # 4. Apply mask. Set zero when invalid input
        features = tf.where(mask,features,tf.zeros_like(features))

        sum = tf.reduce_sum(features, axis=1, keepdims=True) # (B,1,F)
        mean = tf.divide(sum,active_vertices) # (B,1,F)
        mean = tf.tile(mean, [1, self.v, 1]) # (B,V,F)

        output = tf.concat([features, mean], axis=-1) # (B,V,2F)

        if self.kernel_quantizer is not None:
            quantized_output = self.kernel_quantizer(output)
        else:
            quantized_output = output

        return quantized_output 

    def compute_output_shape(self, input_shapes):
        return input_shapes[0][:-1] + (input_shapes[0][-1] * 2,)
    
    def get_quantization_config(self):
        return {
            "kernel_quantizer":
                str(self.kernel_quantizer),
            "divider_quantizer":
                str(self.kernel_quantizer),
            "units" : str(self.units)
        }

    def get_quantizers(self):
        return [self.kernel_quantizer, self.divider_quantizer]