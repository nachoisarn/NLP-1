# Cell 0
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import os

from sklearn.model_selection import train_test_split
from ast import literal_eval
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report

# Cell 1
columna_respuesta = 'respuesta'
clasificaciones=['[UNK]',
 'precios y ofertas',
 'atención y amabilidad del vendedor',
 'variedad de productos y talla',
 'no corresponde al negocio',
 'falta de personal',
 'satisfacción general',
 'otros',
 'orden y limpieza',
 'cambios y devoluciones',
 'descontento general',
 'no aplica']

# Cell 2
newdf = pd.read_excel("new_data.xlsx", names=["respuesta", "sentiment"])
#Dejamos la clasificacion en el formato que nos sirve
newdf["sentiment"] = newdf["sentiment"].apply(lambda x: list(x.lower().split(",")))
newdf

# Cell 3


# Cell 4


# Cell 5
arxiv_data = pd.read_excel("Entrenamiento/data_entrenamiento_Tienda_2022-10-20.xlsx")
arxiv_data["sentiment"] = arxiv_data.sentiment.apply(lambda x: str(x).lower())
arxiv_data

# Cell 6
NCN = arxiv_data.query("Validación == 'No corresponde al negocio' or Validacion_2 == 'No corresponde al negocio' or Validacion_3 == 'No corresponde al negocio'")

# Cell 7
NCN

# Cell 8
NCN_revisado = pd.read_excel("NCN_revisar.xlsx")
NCN_revisado.dropna(inplace=True)
despacho =NCN_revisado["Despacho y retiro"]

# Cell 9
arxiv_data.loc[:,["Validacion_4"]] = arxiv_data.where(~arxiv_data.respuesta.isin(despacho),"Despacho y retiro", axis=0)

arxiv_data["Validacion_4"] = arxiv_data.apply(lambda row:None if isinstance(row["Validacion_4"], int) else row["Validacion_4"], axis=1 )


arxiv_data.loc[arxiv_data.eval("Validacion_4 == 'Despacho y retiro'"), ["sentiment"]] =arxiv_data.loc[arxiv_data.eval("Validacion_4 == 'Despacho y retiro'")].sentiment.apply(
    lambda x: x.replace("no corresponde al negocio","despacho y retiro"))

# Cell 10
arxiv_data.drop(columns=['Unnamed: 0','survey_sending_id','survey_id','company_id','survey_name','negocio','Validación','Validacion_2','Validacion_3','Validacion_4','Validacion_5'], inplace=True)

# Cell 11
def arreglo (lista):
    lista[f"{columna_respuesta}"] = lista[f"{columna_respuesta}"].str.replace('á','a')
    lista[f"{columna_respuesta}"] = lista[f"{columna_respuesta}"].str.replace('é','e')
    lista[f"{columna_respuesta}"] = lista[f"{columna_respuesta}"].str.replace('í','i')
    lista[f"{columna_respuesta}"] = lista[f"{columna_respuesta}"].str.replace('ó','o')
    lista[f"{columna_respuesta}"] = lista[f"{columna_respuesta}"].str.replace('ú','u')
    lista[f"{columna_respuesta}"] = lista[f"{columna_respuesta}"].str.replace('ñ','n')
    #Busca las respuestas que tengan mas de 3 palabras, las que tienen 3 o menos probablemente tienen validacion inconclusa
    lista = lista.loc[lista[f"{columna_respuesta}"].str.len()>3]
    #Vamos a añadir mas items a la limpieza
    caracteres='""#$%&!\?¡¿:;()_´*~{}[]^`=°|¬<>-+/0123456789@'
    caracteres2=',.'   
    
    for caracter in caracteres:
        lista[f"{columna_respuesta}"] = lista[f"{columna_respuesta}"].str.replace(caracter,'')
    for caracter in caracteres2:
        lista[f"{columna_respuesta}"] = lista[f"{columna_respuesta}"].str.replace(caracter, ' ')
        lista[f"{columna_respuesta}"] = lista[f"{columna_respuesta}"].str.lower()
    
    return lista

# Cell 12
arxiv_data = arreglo(arxiv_data);
arxiv_data

# Cell 13


# Cell 14
# Se elimina la data duplicada
arxiv_data = arxiv_data[~arxiv_data[columna_respuesta].duplicated()]
arxiv_data

# Cell 15
# Se deben eliminar los comentarios que tengan un mix de labels únicos, de lo contrario no se puede hacer el split más adelante
# Igual es como rara esta condición
arxiv_data_filtered = arxiv_data.groupby("sentiment").filter(lambda x: len(x) > 1)
print(len(arxiv_data_filtered))

# Cell 16
arxiv_data_filtered["sentiment"] = arxiv_data_filtered["sentiment"].replace(
    "claridad de precios y ofertas","precios y ofertas", regex=True);

# Cell 17
# Esta línea se asgura de que lo que hay dentro de la Serie son strings
arxiv_data_filtered["sentiment"] = arxiv_data_filtered["sentiment"].apply(
    lambda x: literal_eval(x));

# Cell 18
todos_los_datos = pd.concat([arxiv_data_filtered,newdf])
todos_los_datos=shuffle(todos_los_datos)


# Cell 19
todos_los_datos.shape

# Cell 20


# Cell 21
# sacamos 50 datos de cada label y los metemos a newdf
for clase in clasificaciones:
    mask = arxiv_data_filtered.sentiment.apply(lambda x: x == [clase])
    n_de_muestras=arxiv_data_filtered[mask].shape[0]
    if n_de_muestras < 50:
        datos = arxiv_data_filtered[mask].sample(n_de_muestras)
    else:
        datos = arxiv_data_filtered[mask].sample(50)
    newdf = pd.concat([newdf,datos])
mask = arxiv_data_filtered.sentiment.apply(lambda x: len(x)>1)

newdf = pd.concat([newdf,arxiv_data_filtered[mask]])

# Cell 22
# mezclamos los datos de forma aleatoria
newdf=shuffle(newdf)
newdf = newdf.reset_index(drop=True)
with pd.ExcelWriter('DataEntrenamiento.xlsx', engine = 'xlsxwriter') as writer: 
    newdf.to_excel(writer)
    writer.save()


# Cell 23
x =newdf.sentiment.apply(lambda x: str(x)[1:-1]).str.get_dummies(sep=', ').sum().sort_values()
x

# Cell 24
fig, ax = plt.subplots(figsize =(24, 9))
# Horizontal Bar Plot
ax.barh(x.index, x.values)
 
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
 

ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
 
# Add padding between axes and labels
ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 10)
 
# Add x, y gridlines
ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)
 
# Show top values
ax.invert_yaxis()
 
# Add annotation to bars
for i in ax.patches:
    plt.text(i.get_width()+0.2, i.get_y()+0.5,
             str(round((i.get_width()), 2)),
             fontsize = 10, fontweight ='bold',
             color ='grey')
plt.savefig("distribucionEntrenamiento.png")
 

# Cell 25
newdf
#####################################################################################################################################
# Cell 26
# Nos queda claro que las categorias que menos tienen son: "Cambios y Devoluciones":2,
# "Orden y Limpieza":23, "Descontento General":4, "No Aplica":13, "Otros":33,

# Cell 27
test_split=0.2

train_df, test_df = train_test_split(
    arxiv_data_filtered,
    test_size=test_split,
    stratify=arxiv_data_filtered["sentiment"].values,
    # el stratify trata de balancear la data de entrenamiento en cuanto a etiquetas
)
train_df = newdf
# val_df sería una fracción del sample. frac es un numero entre 0 y 1
val_df = test_df.sample(frac=0.3)
#Qué es el frac? cuál es el rango de variación de este numero?
test_df.drop(val_df.index, inplace=True)
print(f"Entrenamiento: {len(train_df)}")
print(f"Validación: {len(val_df)}")
print(f"Test: {len(test_df)}")

# Cell 28
# terms es una lista de labels en formato de tensor
terms = tf.ragged.constant(train_df["sentiment"].values)
# Aquí se setea el modo multi-hot
lookup = tf.keras.layers.StringLookup(output_mode="multi_hot") #multi_hot para multi label
lookup.adapt(terms)
# vocab es la lista de todos los labels por separado
vocab = lookup.get_vocabulary()

def invert_multi_hot(encoded_labels):
    """Reverse a single multi-hot encoded label to a tuple of vocab terms."""
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(vocab, hot_indices)

# Cell 29
train_df["sentiment"].value_counts().head(15)

# Cell 30
train_df[columna_respuesta].apply(lambda x: len(x.split(" "))).describe()

# Cell 31
# Lo que se buscaba con la siguiente fila es determinar una cota para el entrenamiento del modelo.
# Se tomarían solo las primeras 24 palabras de cada respuesta
# (un largo de 24 palabras abarca al 75% de la data)
max_seqlen = int(train_df[columna_respuesta].apply(lambda x: len(x.split(" "))).describe()['75%'])
# bach_size tiene que ver con la capacidad de procesamiento. Determina la cantidad de palabras que el modelo toma de cada respuesta para su entrenamiento.
batch_size = 20

padding_token = "<pad>"
auto = tf.data.AUTOTUNE

# Esta función nos retorna un formato conveniente a trabajar, es decir, el dataset.
def make_dataset(dataframe, is_train=True):
    labels = tf.ragged.constant(dataframe["sentiment"].values)
    label_binarized = lookup(labels).numpy()
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe[columna_respuesta].values, label_binarized)
    )
    dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
    return dataset.batch(batch_size)

def dataset_predict(dataframe):
    labels = tf.ragged.constant(dataframe["sentiment"].values)
    label_binarized = lookup(labels).numpy()
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe[columna_respuesta].values, label_binarized)
    )
    
    return dataset.batch(dataset.cardinality().numpy())

# Cell 32
train_dataset = make_dataset(train_df, is_train=True)
validation_dataset = make_dataset(val_df, is_train=False)
test_dataset = make_dataset(test_df, is_train=False)

# Cell 33


# Cell 34
vocabulary = set()
train_df[columna_respuesta].str.lower().str.split().apply(vocabulary.update)
vocabulary_size = len(vocabulary)

# Cell 35
# Se crea el vectorizado de las palabras. Tiene tantas columnas como palabras distintas
text_vectorizer = layers.TextVectorization(
    # ngrams define la cantidad de palabras que halla de mayor "importancia".
    # no toma palabra por palabra, sino pares de palabras consecutivas.
    max_tokens=vocabulary_size, ngrams=2, output_mode="tf_idf")

# `TextVectorization` layer needs to be adapted as per the vocabulary from our
# training set.

# Este adapt no sabemos bien qué hace, pero resolvió el error de "shapes" que impedía exportar el modelo
with tf.device("/CPU:0"):
    text_vectorizer.adapt(train_dataset.map(lambda text, label: text))

# Se aplica el text_vectorizer
train_dataset = train_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)
validation_dataset = validation_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)
test_dataset = test_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)

# Cell 36
def make_model():
    shallow_mlp_model = keras.Sequential(
        [

            layers.Input(shape=(vocabulary_size)),
            layers.Dense(512, activation="relu"),
            layers.Dropout(.5),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(lookup.vocabulary_size(), activation="sigmoid"),
        ]
    )
    return shallow_mlp_model

# Cell 37
# epochs es la cantidad de iteraciones que cada neurona realiza para ajustar los parámetros
epochs = 50
# el learning_rate determina cuánto "me devuelvo" en el camino de optimización
optimizer = keras.optimizers.Adam(learning_rate=0.0001)

shallow_mlp_model = make_model()
shallow_mlp_model.compile(
    # Esto está adaptado para la naturaleza de NLP
    loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"]
)

history = shallow_mlp_model.fit(
    train_dataset, validation_data=validation_dataset, epochs=epochs
)


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
    

# Se pone en maarcha el entrenamiento
plot_result("loss")
plot_result("binary_accuracy")

# Cell 38
#Evaluando el modelo
_, binary_acc = shallow_mlp_model.evaluate(test_dataset)
print(f"Categorical accuracy on the test set: {round(binary_acc * 100, 2)}%.")

# Cell 39
# Create a model for inference.
model_for_inference = keras.Sequential([text_vectorizer, shallow_mlp_model])

# Create a small dataset just for demoing inference.
inference_dataset = make_dataset(test_df.sample(100), is_train=False)
text_batch, label_batch = next(iter(inference_dataset))
predicted_probabilities = model_for_inference.predict(text_batch)

# Perform inference.
for i, text in enumerate(text_batch[:15]):
    label = label_batch[i].numpy()[None, ...]
    print(f"Abstract: {text}")
    print(f"Label(s): {invert_multi_hot(label[0])}")

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

# Cell 40
inference_dataset1 = dataset_predict(train_df)
text_batch1, label_batch1 = next(iter(inference_dataset1))
predicted_probabilities1 = model_for_inference.predict(text_batch1)

# Cell 41
print(classification_report(label_batch1.numpy().argmax(axis=1), predicted_probabilities1.argmax(axis=1)))

# Cell 42
# En esta celda podemos probar el modelo escribiendo nuestras propias oraciones
textos=["no me llegan nunca los productos"]
df=pd.DataFrame()
df[columna_respuesta]=textos
df["sentiment"]=[[""]]

ejemplo=make_dataset(df, is_train=False)

ejemplo,_=next(iter(ejemplo))
prediccion=model_for_inference.predict(ejemplo)


label_ejemplo = [
        x
        for _, x in sorted(
            zip(prediccion[0], lookup.get_vocabulary()),
            key=lambda pair: pair[0],
            reverse=True,
        ) if _ > 0.4
    ]
if len(label_ejemplo) == 0:
    label_ejemplo = [
    x
    for _, x in sorted(
        zip(prediccion[0], lookup.get_vocabulary()),
        key=lambda pair: pair[0],
        reverse=True,
    )][:1]
label_ejemplo

# Cell 43


# Cell 44
#creamos el dataframe
dtypes = np.dtype(
    [
        ("texto", str),
        ("label", np.ndarray),
        ("prediccion", list)
    ]
)

df_predicted = pd.DataFrame(np.empty(0, dtype=dtypes))

# Cell 45
# Create a model for inference.
model_for_inference = keras.Sequential([text_vectorizer, shallow_mlp_model])

# Create a small dataset just for demoing inference.
inference_dataset = dataset_predict(todos_los_datos)
text_batch, label_batch = next(iter(inference_dataset))
predicted_probabilities = model_for_inference.predict(text_batch)
# Perform inference.
for i, text in enumerate(text_batch):
    label = label_batch[i].numpy()[None, ...]

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
        
    df_predicted.loc[len(df_predicted.index)] = [text.numpy(),invert_multi_hot(label[0]),top_labels]
   

# Cell 46
df_predicted.shape

# Cell 47

with pd.ExcelWriter('predicciones.xlsx', engine = 'xlsxwriter') as writer: 
    workbook = writer.book
    df_predicted.to_excel(writer, sheet_name='Predicciones')
    worksheet = workbook.add_worksheet("Graficos") 

    worksheet.insert_image('A5','loss_plot.png')
    worksheet.insert_image('K5','binary_accuracy_plot.png')
    
    
    worksheet.write_comment("A1","Grafico Loss/Epoch",  {"visible": True})
                                        
    worksheet.write_comment("K1","Grafico Accuracy/Epoch", {"visible": True})
    writer.save()

# Cell 48
import pickle
pickle.dump({'config': text_vectorizer.get_config(),
             'weights': text_vectorizer.get_weights()}
            , open("tv_layer.pkl", "wb"))

# Cell 49
#ESTO SOLO CORRE SI SE QUIERE GUARDAR NUEVAMENTE EL MODELO

filename = 'train_df.sav'
pickle.dump(train_df, open(filename, 'wb'))

# Cell 50
#ESTO SOLO CORRE SI SE QUIERE GUARDAR NUEVAMENTE EL MODELO
!mkdir -p saved_model
shallow_mlp_model.save('saved_model/my_model')

# Cell 51
#ESTO SOLO CORRE SI SE QUIERE GUARDAR NUEVAMENTE EL MODELO
filename = 'vocabulary_size.sav'
pickle.dump(vocabulary_size, open(filename, 'wb'))

# Cell 52
lookup.get_vocabulary()

# Cell 53


