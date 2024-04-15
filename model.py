from tensorflow.keras.models import model_from_json
import numpy as np
import tensorflow as tf

class ExerciseModel(object):
    exercise = ['Push-Up', 'Pull-up', 'sit-up', 'Jumping Jack', 'Squat']

    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, 'r') as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()

    def predict_exercise(self, inp):
        self.preds = self.loaded_model.predict(inp)
        return ExerciseModel.exercise[np.argmax(self.preds)]