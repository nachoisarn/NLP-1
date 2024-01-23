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



def eval_model(modelpath, filepath):

    train_dataset, validation_dataset, test_dataset = datos(filepath)
    model = tf.keras.models.load_model(modelpath)

    _, binary_acc = model.evaluate(test_dataset)
    print(f"Categorical accuracy on the test set: {round(binary_acc * 100, 2)}%.")


def main():
    model = eval_model('trained_model.h5', 'data_entrenamiento.csv')


if __name__ == "__main__":
    main()