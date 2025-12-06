from tensorflow import keras
from tensorflow.keras import layers

# Put the Dropout layer just before the layer you want the dropout applied to:
keras.Sequential([
    # ...
    layers.Dropout(rate=0.3), # apply 30% dropout to the next layer
    layers.Dense(16),
    # ...
])

# It seems that batch normalization can be used at almost any point in a network. You can put it after a layer...
keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.BatchNormalization(),
])
# ... or between a layer and its activation function:
keras.Sequential([
    layers.Dense(16),
    layers.BatchNormalization(),
    layers.Activation('relu'),
])
# And if you add it as the first layer of your network it can act as a kind of adaptive preprocessor, 
# standing in for something like Sci-Kit Learn's StandardScaler. 