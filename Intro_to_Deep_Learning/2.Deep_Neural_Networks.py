from tensorflow.python import keras
from tensorflow.python.keras import layers
import pandas as pd

'''
model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),
    # the linear output layer 
    layers.Dense(units=1),
])
'''

concrete = pd.read_csv('Intro_to_Deep_Learning/datasets/concrete.csv')
print(concrete.head())

input_shape = [8]

model = keras.Sequential([
    layers.Dense(units=512, activation='relu', input_shape=input_shape),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=1),
])
'''
    layers.Dense(units=8),
    layers.Activation('relu')
# This is completely equivalent to the ordinary way: 
    layers.Dense(units=8, activation='relu')
'''
