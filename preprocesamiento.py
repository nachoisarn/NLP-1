import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlsxwriter

import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from ast import literal_eval






def load_and_preprocess_data(filepath1,filepath2,filepath3):

    def proceso_caracteres(df):
        columna_respuesta = 'respuesta'
        
        # Reemplazar caracteres especiales con una expresión regular
        caracteres_especiales = r'[áéíóúñ]'
        df.loc[:,f"{columna_respuesta}"] = df[f"{columna_respuesta}"].str.replace(caracteres_especiales, lambda m: m.group(0).normalize('NFKD').encode('ASCII', 'ignore').decode('utf-8'), regex=True)
        
        # Filtrar respuestas con más de 3 palabras
        df = df[df[f"{columna_respuesta}"].str.split().str.len() > 3]

        # Reemplazar caracteres especiales y convertir a minúsculas
        caracteres = r'[#$%&!\?¡¿:;()_´*~{}[]^`=°|¬<>-+/0123456789@,.]'
        df.loc[:,f"{columna_respuesta}"] = df[f"{columna_respuesta}"].str.replace(caracteres, ' ', regex=True)
        df.loc[:,f"{columna_respuesta}"] = df[f"{columna_respuesta}"].str.lower()
        
        return df


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


    # Leer el archivo Excel y especificar el nombre de las columnas
    newdf = pd.read_excel(filepath1, names=["respuesta", "sentiment"])

    # Convertir las palabras en minúsculas y dividirlas en una lista
    newdf.loc[:,"sentiment"] = newdf["sentiment"].str.lower().str.split(",")

    # Leer el archivo Excel
    arxiv_data = pd.read_excel(filepath2)

    # Convertir la columna 'sentiment' a minúsculas
    arxiv_data.loc[:,"sentiment"] = arxiv_data["sentiment"].str.lower()

    # Filtrar las filas donde alguna columna de Validación es igual a 'No corresponde al negocio'
    NCN = arxiv_data[(arxiv_data['Validación'] == 'No corresponde al negocio') | (arxiv_data['Validacion_2'] == 'No corresponde al negocio') | (arxiv_data['Validacion_3'] == 'No corresponde al negocio')]

    # Leer el archivo "NCN_revisar.xlsx" y eliminar filas con valores nulos
    NCN_revisado = pd.read_excel(filepath3).dropna()

    # Extraer la columna "Despacho y retiro" en la variable despacho
    despacho = NCN_revisado["Despacho y retiro"]


    # Asignar 'Despacho y retiro' a la columna 'Validacion_4' donde 'respuesta' no está en 'despacho'
    arxiv_data.loc[:,["Validacion_4"]] = arxiv_data.where(~arxiv_data.respuesta.isin(despacho), "Despacho y retiro", axis=0)

    # Reemplazar valores 'int' en la columna 'Validacion_4' con 'None'
    arxiv_data["Validacion_4"] = arxiv_data.apply(lambda row: None if isinstance(row["Validacion_4"], int) else row["Validacion_4"], axis=1)

    # Reemplazar valores en la columna 'sentiment' donde 'Validacion_4' es igual a 'Despacho y retiro'
    arxiv_data.loc[arxiv_data.eval("Validacion_4 == 'Despacho y retiro'"), ["sentiment"]] = arxiv_data.loc[arxiv_data.eval("Validacion_4 == 'Despacho y retiro'")].sentiment.apply(
        lambda x: x.replace("no corresponde al negocio", "despacho y retiro"))

    # Eliminar columnas no necesarias
    arxiv_data.drop(columns=['Unnamed: 0', 'survey_sending_id', 'survey_id', 'company_id', 'survey_name', 'negocio', 'Validación', 'Validacion_2', 'Validacion_3', 'Validacion_4', 'Validacion_5'], inplace=True)

    # Limpiar texto
    arxiv_data = proceso_caracteres(arxiv_data);

    # Eliminar filas duplicadas en el DataFrame basado en la columna 'columna_respuesta'
    arxiv_data = arxiv_data[~arxiv_data[columna_respuesta].duplicated()]

    # Filtrar las filas donde la columna 'sentiment' tiene mezcla de etiquetas únicas
    # Esto se hace para asegurarse de que las etiquetas sean coherentes antes de realizar divisiones posteriores
    arxiv_data_filtered = arxiv_data.groupby("sentiment").filter(lambda x: len(x) > 1)

    # Reemplazar una etiqueta específica en la columna 'sentiment'
    arxiv_data_filtered["sentiment"] = arxiv_data_filtered["sentiment"].replace(
        "claridad de precios y ofertas", "precios y ofertas", regex=True);


    # Asegurarse de que los valores en la columna 'sentiment' sean interpretados como listas
    arxiv_data_filtered["sentiment"] = arxiv_data_filtered["sentiment"].apply(
        lambda x: literal_eval(x));

    # Concatenar 'arxiv_data_filtered' con 'newdf' y luego mezclar los datos de forma aleatoria
    todos_los_datos = pd.concat([arxiv_data_filtered, newdf])
    todos_los_datos = shuffle(todos_los_datos)


    # Crear una función para muestrear los datos
    def sample_data(data):
        n_de_muestras = min(data.shape[0], 50)
        return data.sample(n_de_muestras)

    newdf = pd.DataFrame()  # DataFrame vacío para almacenar los resultados

    # # Añadir una columna para marcar los registros seleccionados
    arxiv_data_filtered['seleccionado'] = False

    for clase in clasificaciones:
        # Filtrar y muestrear datos
        filtered_data = arxiv_data_filtered[(arxiv_data_filtered['sentiment'].apply(lambda x: x == [clase])) & (~arxiv_data_filtered['seleccionado'])]
        sampled_data = sample_data(filtered_data)
        
        # Marcar los registros seleccionados
        arxiv_data_filtered.loc[sampled_data.index, 'seleccionado'] = True
        
        # Concatenar con newdf
        newdf = pd.concat([newdf, sampled_data])

    # Filtrar para registros multi-etiqueta que no han sido seleccionados
    mask = (arxiv_data_filtered.sentiment.apply(lambda x: len(x) > 1)) & (~arxiv_data_filtered['seleccionado'])
    multi_label_data = arxiv_data_filtered[mask]
    newdf = pd.concat([newdf, multi_label_data])

    # Mezclar y reiniciar índice
    newdf = shuffle(newdf).reset_index(drop=True)

    # Opcional: Eliminar la columna 'seleccionado'
    del arxiv_data_filtered['seleccionado'], newdf['seleccionado']


    # Guardar newdf como archivo CSV
    newdf.to_csv('DataEntrenamiento.csv', index=False)  # No incluir el índice en el archivo CSV

    return newdf


def main():
    data = load_and_preprocess_data('new_data.xlsx',
                                                'data_entrenamiento_Tienda_2022-10-20.xlsx',
                                                'NCN_revisar.xlsx')

    # Guardar los DataFrames procesados para su uso posterior
    data.to_csv('data_entrenamiento.csv', index=False)


if __name__ == "__main__":
    main()




'''
###############################################################################################################
# Contar las etiquetas en la columna 'sentiment'
x = newdf.sentiment.apply(lambda x: str(x)[1:-1]).str.get_dummies(sep=', ').sum().sort_values()


# Crear una figura y un eje
fig, ax = plt.subplots(figsize=(24, 9))

# Crear un gráfico de barras horizontales
ax.barh(x.index, x.values)

# Eliminar los bordes de los ejes
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)

# Configurar la posición de las marcas de los ejes
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Agregar espaciado entre los ejes y las etiquetas
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

# Agregar líneas de cuadrícula
ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)

# Invertir el eje y para mostrar las etiquetas más comunes en la parte superior
ax.invert_yaxis()

# Agregar anotaciones a las barras
for i in ax.patches:
    plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
             str(round((i.get_width()), 2)),
             fontsize=10, fontweight='bold', color='grey')
#plt.show()
# Guardar el gráfico como imagen
plt.savefig("distribucionEntrenamiento.png")

'''