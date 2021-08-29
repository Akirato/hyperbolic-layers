import tensorflow as tf
from tensorflow import keras
from .lin_hyp import LinearHyperbolic
from .att_hyp import AttentionHyperbolic

class AggregationHyperbolic(keras.layers.Layer):

    def __init__(self,units, manifold, c, activation, use_bias=True, use_att=False, local_agg=True):
        super().__init__()
        self.c = tf.Variable([c], dtype="float64")
        self.manifold = manifold
        self.use_bias = use_bias
        self.use_att = use_att
        self.local_agg = local_agg
        if self.use_att:
            self.att = AttentionHyperbolic(units, manifold, c, activation, use_bias)

    def build(self, batch_input_shape):
        super().build(batch_input_shape)  # must be at the end

    def call(self, inputs, adj):
        """
        Called during forward pass of a neural network. Uses hyperbolic matrix multiplication
        """
        inputs = tf.cast(inputs, tf.float64)
        adj = tf.cast(adj, tf.float64)

        inputs_tangent = self.manifold.logmap0(inputs)
        if self.use_att:
            if self.local_agg:
                inputs_local_tangent = []
                for i in range(inputs.shape[1]):
                    inputs_local_tangent.append(self.manifold.logmap(inputs[:,i], inputs))
                inputs_local_tangent = tf.stack(inputs_local_tangent, axis=1)
                adj_att = self.att(inputs_tangent,adj)
                att_rep = tf.expand_dims(adj_att,-1) * inputs_local_tangent
                support_t = tf.reduce_sum(att_rep, axis=1)
                output = self.manifold.proj(self.manifold.expmap(inputs, support_t))
                return output
            else:
                adj_att = self.att(inputs_tangent,adj)
                support_t = self.manifold.mobius_matvec(adj_att, inputs_tangent)
        else:
            support_t = self.manifold.mobius_matvec(adj, inputs_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t))
        return output

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "manifold": self.manifold,
            "curvature": self.c
        }

class ActivationHyperbolic(keras.layers.Layer):
    """
    Hyperbolic activation layer.
    """
    def __init__(self, manifold, c, activation=None, use_bias=True, use_att=False, local_agg=True):
        super().__init__()
        self.c = tf.Variable([c], dtype="float64")
        self.manifold = manifold
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.use_att = use_att
        self.local_agg = local_agg

    def build(self, batch_input_shape):
        super().build(batch_input_shape)  # must be at the end
    
    def call(self, inputs):
        """
        Called during forward pass of a neural network. Uses hyperbolic matrix multiplication
        """
        inputs = tf.cast(inputs, tf.float64)
        inputs_t = self.activation(self.manifold.logmap0(inputs))
        inputs_t = self.manifold.proj_tan0(inputs_t)
        return self.manifold.proj(self.manifold.expmap0(inputs_t))

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "activation": keras.activations.serialize(self.activation),
            "manifold": self.manifold,
            "curvature": self.c
        }
        

class ConvolutionHyperbolic(keras.layers.Layer):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, c, activation=None, use_bias=True, use_att=True, local_agg=True):
        super(ConvolutionHyperbolic, self).__init__()
        self.c = tf.Variable([c], dtype="float64")
        self.manifold = manifold
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.use_att = use_att
        self.local_agg = local_agg
        self.hyp_act = ActivationHyperbolic(manifold, self.c, self.activation)

    def build(self, batch_input_shape):
        units = batch_input_shape[-1]
        self.linear = LinearHyperbolic(units, self.manifold, self.c, self.activation, self.use_bias)
        self.agg = AggregationHyperbolic(units, self.manifold, self.c, self.activation, self.use_bias, self.use_att, self.local_agg)
        super().build(batch_input_shape)  # must be at the end

    def call(self, input, adj):
        h = self.linear(input)
        h = self.agg(h, adj)
        h = self.hyp_act(h)
        return h