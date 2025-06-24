import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
from td_utils import *

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model, model_from_json
from tensorflow.keras.layers import (
    Dense, Activation, Dropout, Input, TimeDistributed, 
    Conv1D, GRU, BatchNormalization
)
from tensorflow.keras.optimizers import Adam

# Constants
Tx = 5511      # Input time steps (spectrogram width)
n_freq = 101   # Frequency bins (spectrogram height)
Ty = 1375      # Output time steps (label vector length)

# Load training data
X = np.load("./XY_train/X0.npy")
Y = np.load("./XY_train/Y0.npy")
X = np.concatenate((X, np.load("./XY_train/X1.npy")), axis=0)
Y = np.concatenate((Y, np.load("./XY_train/Y1.npy")), axis=0)
Y = np.swapaxes(Y, 1, 2)

# Load dev set
X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")

# Define the model
def modelf(input_shape):
    X_input = Input(shape=input_shape)

    X = Conv1D(196, kernel_size=15, strides=4)(X_input)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Dropout(0.8)(X)

    X = GRU(128, return_sequences=True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)

    X = GRU(128, return_sequences=True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.8)(X)

    X = TimeDistributed(Dense(1, activation="sigmoid"))(X)

    return Model(inputs=X_input, outputs=X)

# Instantiate and compile model
model = modelf(input_shape=(Tx, n_freq))
model.summary()

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])

# Train the model
model.fit(X, Y, batch_size=20, epochs=100)

# Evaluate on dev set
loss, acc = model.evaluate(X_dev, Y_dev)
print("Dev set accuracy =", acc)

# Load pre-trained model from JSON + weights (optional override)
with open('./models/model_new3.json', 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights('./models/model_new3.h5')
