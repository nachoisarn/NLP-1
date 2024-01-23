import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.layers import TextVectorization
import tensorflow as tf

filepath = 'data_entrenamiento.csv'

def split(filepath = filepath):
    from ast import literal_eval
    # Cargar los datos
    training_data = pd.read_csv(filepath, converters={'sentiment': literal_eval})
    

    # Dividir los datos en conjuntos de entrenamiento, validación y prueba
    test_split = 0.2
    train_dataframe, test_dataframe = train_test_split(
        training_data,
        test_size=test_split,
        random_state=0,
        stratify=training_data["sentiment"].values)  # Asegurar distribución balanceada
    
 
    # Dividir más para obtener un conjunto de validación
    val_dataframe = test_dataframe.sample(frac=0.3, random_state = 0)
    test_dataframe.drop(val_dataframe.index, inplace=True)

    return train_dataframe, test_dataframe, val_dataframe


def vocabulary_fun(filepath = filepath):
    #datos
    train_dataframe, test_dataframe, val_dataframe = split(filepath)

    # Paso 1: Creación del Vocabulario
    # Inicializar un conjunto para almacenar el vocabulario único
    vocabulary = set()
    # Actualizar el conjunto de vocabulario con palabras únicas de las respuestas
    # Convertir a minúsculas y dividir cada respuesta en palabras
    train_dataframe['respuesta'].str.lower().str.split().apply(vocabulary.update)

    # Paso 2: Cálculo del Tamaño del Vocabulario
    # Calcular el tamaño del vocabulario basado en las palabras únicas
    vocabulary_size = len(vocabulary)

    return vocabulary

def lookup_fun(filepath = filepath):
 
    from ast import literal_eval
    train_dataframe, test_dataframe, val_dataframe = split(filepath)
    
    unique_labels = set([elemento for lista in train_dataframe['sentiment'] for elemento in lista])

    # Crear y adaptar la capa StringLookup
    lookup = keras.layers.StringLookup(output_mode="multi_hot")
    lookup.adapt(list(unique_labels))
    vocab = lookup.get_vocabulary()
    
    return lookup


def invert_multi_hot(vocab, encoded_labels):
        import numpy as np
        """Reverse a single multi-hot encoded label to a tuple of vocab terms."""
        hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
        return np.take(vocab, hot_indices)

def textVectorizer(filepath = filepath):
    from tensorflow import keras
    from keras import layers

    dataframe, _, _ = split(filepath)

    vocabulary = vocabulary_fun(filepath)
    vocabulary_size = len(vocabulary)
    # Paso 3: Configuración de la Capa TextVectorization
    # Inicializar la capa TextVectorization de TensorFlow
    # Configurar para usar bigramas y TF-Idataframe como método de vectorización
    text_vectorizer = layers.TextVectorization(
        max_tokens=vocabulary_size,  # Establecer el número máximo de tokens
        ngrams=2,                    # Considerar bigramas
        output_mode="tf_idf")         # Utilizar TF-IDF para la vectorización
    
    # Paso 4: Adaptación de la Capa TextVectorization
    # Adaptar la capa al conjunto de datos de entrenamiento
    # Realizar esta operación en la CPU para evitar problemas de memoria en la GPU
    with tf.device("/CPU:0"):
        # Convertir la columna 'respuesta' a un tensor antes de adaptar
        respuestas = tf.convert_to_tensor(dataframe['respuesta'].values, dtype=tf.string)
        text_vectorizer.adapt(respuestas)

    return text_vectorizer
    # Paso 5: Aplicación de la Vectorización al Dataset
    # Definir una función para aplicar la vectorización a los datasets
def make_dataset(dataframe, lookup, batch_size, is_train=True):
        import numpy as np
        # Función para binarizar las etiquetas de cada muestra
        def binarize_labels(labels):
            return lookup(np.array(labels)).numpy()

        # Aplicar la función de binarización a cada conjunto de etiquetas en la columna 'sentiment'
        label_binarized = np.array(dataframe['sentiment'].apply(binarize_labels).to_list())

        # Convertir las respuestas a tensores
        features = tf.convert_to_tensor(dataframe['respuesta'].values, dtype=tf.string)

        # Verificar la correspondencia de dimensiones
        if features.shape[0] != label_binarized.shape[0]:
            raise ValueError("El número de respuestas no coincide con el número de etiquetas binarizadas.")

        # Crear el dataset
        dataset = tf.data.Dataset.from_tensor_slices((features, label_binarized))

        # Mezclar y agrupar en lotes si es para entrenamiento
        if is_train:
            dataset = dataset.shuffle(buffer_size=len(dataframe))
        return dataset.batch(batch_size)

def dataset_predict(dataframe, lookup):
        import numpy as np
        # Función para binarizar las etiquetas de cada muestra
        def binarize_labels(labels):
            return lookup(np.array(labels)).numpy()

        # Aplicar la función de binarización a cada conjunto de etiquetas en la columna 'sentiment'
        label_binarized = np.array(dataframe['sentiment'].apply(binarize_labels).to_list())

        # Convertir las respuestas a tensores
        features = tf.convert_to_tensor(dataframe['respuesta'].values, dtype=tf.string)

        # Verificar la correspondencia de dimensiones
        if features.shape[0] != label_binarized.shape[0]:
            raise ValueError("El número de respuestas no coincide con el número de etiquetas binarizadas.")

        # Crear el dataset
        dataset = tf.data.Dataset.from_tensor_slices((features, label_binarized))

        cardinality = tf.data.experimental.cardinality(dataset).numpy()
        return dataset.batch(cardinality)