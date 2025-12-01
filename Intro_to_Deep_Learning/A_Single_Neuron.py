from tensorflow.python import keras
from tensorflow.python.keras import layers
import pandas as pd

# Create a network with 1 linear unit
'''
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])
'''

red_wine = pd.read_csv('Intro_to_Deep_Learning/datasets/red-wine.csv')
print(red_wine.head())
print(red_wine.shape) # (rows, columns)

input_shape = [11]

model = keras.Sequential([
    layers.Dense(units=1, input_shape=input_shape)
])

w, b = model.weights
print("Weights\n{}\n\nBias\n{}".format(w, b))