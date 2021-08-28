import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input
from tf_hnn.layers.lin_hyp import LinearHyperbolic
from tf_hnn.optimizers.rsgd import RSGD
from tf_hnn.manifolds.poincare import PoincareBall

# Create layers
hyperbolic_layer_1 = LinearHyperbolic(32, PoincareBall(), 1)
hyperbolic_layer_2 = LinearHyperbolic(32, PoincareBall(), 1)
output_layer = LinearHyperbolic(10, PoincareBall(), 1)

# Create optimizer
optimizer = RSGD(learning_rate=0.1)

# Create model architecture
model = tf.keras.models.Sequential([
  Input(shape=(128,)),
  hyperbolic_layer_1,
  hyperbolic_layer_2,
  output_layer
])

# Compile the model with the Riemannian optimizer            
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
print(model.summary())
