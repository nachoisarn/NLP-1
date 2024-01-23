import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from ast import literal_eval

from funciones import split, vocabulary_fun, lookup_fun
from datos import datos




def train_model(modelpath, filepath):

    train_dataset, validation_dataset, test_dataset = datos(filepath)
    
    model = tf.keras.models.load_model(modelpath)
    epochs = 5
    history = model.fit(train_dataset,
                        validation_data=validation_dataset,
                        epochs=epochs)

    def plot_result(item):
        plt.plot(history.history[item], label=item)
        plt.plot(history.history["val_" + item], label="val_" + item)
        plt.xlabel("Epochs")
        plt.ylabel(item)
        plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
        plt.legend()
        plt.grid()
        plt.savefig('{}_plot.png'.format(item))
        plt.show()
        

    # Se pone en marcha el entrenamiento
    plot_result("loss")
    plot_result("binary_accuracy")

    return model

def main():
    model = train_model('model_architecture.h5', 'data_entrenamiento.csv')

    # Guardar la arquitectura del modelo
    model.save('trained_model.h5')

if __name__ == "__main__":
    main()

