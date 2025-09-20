import os
import keras

def load_model(path : str):
    return keras.models.load_model(os.path.abspath(path))