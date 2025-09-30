import os
import keras

def load_model(path : str):
    return keras.models.load_model(os.path.abspath(path))

def predict(model_path : str, img_path: str ):
    model = load_model(path)
    return keras.models.load_model(os.path.abspath(path))