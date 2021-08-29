import tensorflow as tf
from tensorflow import keras


class AttentionHyperbolic(keras.layers.Layer):
    """
    Implementation of a hyperbolic linear layer for a neural network, that inherits from the keras Layer class
    """

    def __init__(self, units, manifold, c, activation=None, use_bias=True):
        super().__init__()
        self.units = units
        self.c = tf.Variable([c], dtype="float64")
        self.manifold = manifold
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, batch_input_shape):
        w_init = tf.random_normal_initializer()
        self.kernel = tf.Variable(
            initial_value=w_init(shape=(batch_input_shape[-1], self.units), dtype="float64"), dtype="float64", trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units),
                initializer="zeros",
                dtype=tf.float64,
                trainable=True,
            )

        super().build(batch_input_shape)  # must be at the end

    def call(self, inputs, adj):
        """
        Called during forward pass of a neural network. Uses hyperbolic matrix multiplication
        """
        # TODO: remove casting and instead recommend setting default tfd values to float64
        inputs = tf.cast(inputs, tf.float64)
        adj = tf.cast(adj, tf.float64)
        
        n = inputs.shape[1]
        inputs_left = tf.expand_dims(inputs,1)
        inputs_left = tf.broadcast_to(inputs_left,(-1,n,-1))

        inputs_right = tf.expand_dims(inputs,1)
        inputs_right = tf.broadcast_to(inputs_right,(n, -1, -1))

        inputs_cat = tf.concat([inputs_left, inputs_right], axis=2)
        att_adj = tf.squeeze(keras.activations.linear(inputs_cat))
        att_adj = keras.activations.sigmoid(att_adj)
        att_adj = tf.matmul(tf.transpose(adj,perm=[0,2,1]), att_adj)
        mv = self.manifold.mobius_matvec(self.kernel, inputs)
        res = self.manifold.proj(mv)
        
        if self.use_bias:
            hyp_bias = self.manifold.expmap0(self.bias)
            hyp_bias = self.manifold.proj(hyp_bias)
            res = self.manifold.mobius_add(res, hyp_bias)
            res = self.manifold.proj(res)

        return self.activation(res)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "units": self.units,
            "activation": keras.activations.serialize(self.activation),
            "manifold": self.manifold,
            "curvature": self.c
        }
