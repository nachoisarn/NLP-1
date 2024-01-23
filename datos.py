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

from funciones import split, vocabulary_fun, lookup_fun, textVectorizer, make_dataset


def datos(filepath):
    dataframe, test_dataframe, val_dataframe = split(filepath)
    
    lookup = lookup_fun(filepath)
    vocab = lookup.get_vocabulary()

    batch_size = 20

    train_dataset = make_dataset(dataframe, lookup, batch_size, is_train=True)
    validation_dataset = make_dataset(val_dataframe, lookup, batch_size, is_train=False)
    test_dataset = make_dataset(test_dataframe, lookup, batch_size, is_train=False)


    vocabulary = vocabulary_fun(filepath)
    vocabulary_size = len(vocabulary)
    # Paso 5: Aplicación de la Vectorización al Dataset
    # Definir una función para aplicar la vectorización a los datasets
        
    text_vectorizer = textVectorizer()

    def vectorize_text(text, label):
        text_vectorized = text_vectorizer(text)
        return text_vectorized, label

    # Aplicar la vectorización a los datasets de entrenamiento, validación y prueba
    # Utilizar operaciones paralelas para mejorar el rendimiento y prefetching para la carga eficiente de datos
    train_dataset = train_dataset.map(vectorize_text, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.map(vectorize_text, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(vectorize_text, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset