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
from keras.models import Sequential

from funciones import split, vocabulary_fun, lookup_fun



def create_model(filepath):
    
    dataframe, test_dataframe, val_dataframe = split(filepath)

    print(f"Entrenamiento: {len(dataframe)}")
    print(f"Validación: {len(val_dataframe)}")
    print(f"Test: {len(test_dataframe)}")

    # Preparación de etiquetas para TensorFlow
    # Asegúrate de que 'sentiment' está en un formato que pueda ser procesado
    # Aquí se asume que 'sentiment' ya está en un formato adecuado

    # Aplanar todas las listas en la columna 'sentiment' para crear un conjunto único de etiquetas
    lookup = lookup_fun(filepath)
    vocab = lookup.get_vocabulary()

    # Establecer el tamaño del lote para el entrenamiento del modelo.
    # El tamaño del lote afecta la cantidad de datos que el modelo procesa en cada iteración durante el entrenamiento.


    # Comentarios adicionales:
    # - max_seqlen es calculado para establecer un límite superior en la longitud de las secuencias de texto, lo cual es necesario para
    #   la mayoría de los modelos de procesamiento de lenguaje natural (NLP).
    # - La elección de usar el percentil 75 es un equilibrio entre capturar suficiente información y no hacer que el modelo sea
    #   innecesariamente complejo o lento al procesar secuencias muy largas.
    # Paso 1: Creación del Vocabulario
    # Inicializar un conjunto para almacenar el vocabulario único
    vocabulary = vocabulary_fun(filepath)
    vocabulary_size = len(vocabulary)
  
    # el learning_rate determina cuánto "me devuelvo" en el camino de optimización
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    def make_model(vocabulary_size, num_classes):
        # Crear un modelo secuencial de Keras.
        shallow_mlp_model = keras.Sequential([
            # Capa de entrada que especifica el tamaño del vocabulario.
            layers.Input(shape=(vocabulary_size,)),

            # Primera capa densa con 512 neuronas y activación ReLU.
            layers.Dense(512, activation="relu"),

            # Capa de Dropout para reducir el sobreajuste.
            layers.Dropout(0.5),

            # Segunda capa densa con 256 neuronas y activación ReLU.
            layers.Dense(256, activation="relu"),

            # Otra capa de Dropout.
            layers.Dropout(0.4),

            # Capa de salida con una neurona por clase y activación sigmoide.
            # La activación sigmoide es apropiada para la clasificación binaria o multietiqueta.
            layers.Dense(num_classes, activation="sigmoid"),
        ])

        return shallow_mlp_model

    # Ejemplo de uso:
    model = make_model(vocabulary_size, lookup.vocabulary_size())
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,
                  metrics=["binary_accuracy"])
    

    # ES MEJOR HACER EL PROCESAMIENTO DEL LOOKUP Y DATA A TENSORFLOW JUNTO CON LA CREACION DEL MODELO PARA NO TENER PROBLEMA
    # AL GUARDARLO
    return model

def main():
    model = create_model('data_entrenamiento.csv')

    # Guardar la arquitectura del modelo
    model.save('model_architecture.h5')

if __name__ == "__main__":
    main()