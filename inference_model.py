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

from funciones import split, vocabulary_fun, lookup_fun, textVectorizer, make_dataset, invert_multi_hot
from datos import datos

def model_of_inference(modelpath):

    #creamos el dataframe
    dtypes = np.dtype(
        [
            ("texto", str),
            ("label", np.ndarray),
            ("prediccion", list)
        ]
    )

    df_predicted = pd.DataFrame(np.empty(0, dtype=dtypes))

    train_dataframe, test_dataframe, val_dataframe = split()
    lookup = lookup_fun()
    vocab = lookup.get_vocabulary()
    # Importamos el modelo
    model = tf.keras.models.load_model(modelpath)
    text_vectorizer = textVectorizer()

    model_for_inference = keras.Sequential([text_vectorizer, model])



    # Create a small dataset just for demoing inference.
    inference_dataset = make_dataset(test_dataframe, lookup, batch_size= 76, is_train=False)
    text_batch, label_batch = next(iter(inference_dataset))
    predicted_probabilities = model_for_inference.predict(text_batch)

    # Perform inference.
    for i, text in enumerate(text_batch[:15]):
        label = label_batch[i].numpy()[None, ...]
        print(f"Abstract: {text}")
        print(f"Label(s): {invert_multi_hot(vocab, label[0])}")

        #predicted_proba = [proba for proba in predicted_probabilities[i]]
        top_labels = [
            x
            for _, x in sorted(
                zip(predicted_probabilities[i], lookup.get_vocabulary()),
                key=lambda pair: pair[0],
                reverse=True,
            ) if _ > 0.4
        ]
        if len(top_labels) == 0:
            top_labels = [
            x
            for _, x in sorted(
                zip(predicted_probabilities[i], lookup.get_vocabulary()),
                key=lambda pair: pair[0],
                reverse=True,
            )][:1]
        print(f"Predicted Label(s): ({', '.join([label for label in top_labels])})")
        print(" ")
        

def main():

    model_of_inference('trained_model.h5')

if __name__ == '__main__':
    main()