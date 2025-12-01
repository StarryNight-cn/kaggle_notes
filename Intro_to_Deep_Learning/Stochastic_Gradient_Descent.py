from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split

fuel = pd.read_csv('Intro_to_Deep_Learning/datasets/fuel.csv')
X = fuel.copy()
y = X.pop('FE')

preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse_output=False),
     make_column_selector(dtype_include=np.object_)),
)

X = preprocessor.fit_transform(X)
y = np.log(y) # log transform target instead of standardizing


print(fuel.head())

input_shape = [X.shape[1]]
model = keras.Sequential([
    keras.Input(shape=input_shape),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),    
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])

# Key point: Using 'adam' optimizer and 'mae' loss function
model.compile(
    optimizer="adam",
    loss="mae",
)

# Key point: Using batch size of 128 and training for 200 epochs
history = model.fit(
    X, y,
    # validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=200,
)

history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5. You can change this to get a different view.
history_df.loc[5:, ['loss']].plot()
plt.show()
