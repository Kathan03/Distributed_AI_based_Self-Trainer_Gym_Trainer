import numpy as np
from keras.models import load_model
#from keras.saving import load_model
import tensorflow as tf


class ExerciseModel(object):
    exercise = ['Push-Up', 'Pull-up', 'sit-up', 'Jumping Jack', 'Squat']

    def __init__(self, model_file):
        self.loaded_model = load_model(model_file)

    def predict_exercise(self, inp):
        self.preds = self.loaded_model.predict(inp)
        return ExerciseModel.exercise[np.argmax(self.preds)]