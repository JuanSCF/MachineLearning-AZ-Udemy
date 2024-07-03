#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

# Redes Neuronales Artificiales


# Parte 1 - Preprocesado de datos

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   
    remainder='passthrough'                        
)
X = onehotencoder.fit_transform(X)
X = X[:, 1:] # evitamos la trampa de las variables dummy. no es necesario para genero

'''
# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
# print(X)
# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# print(X)
'''

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Parte 2 - Construir la RNA

import keras
from keras.models import Sequential # para inicializar los parametros de la red neuronal
from keras.layers import Dense # sirve para declarar y construir cada capa de la red neuronal

# Inicializar la RNA. 
# definir la arquitectura de la red. secuencia de capas o como se relacionan los grafos
classifier = Sequential()

# Añadir las capas de entrada y primera capa oculta
# nro de nodos = nro de variables dependientes
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# units = nro de nodos de la capa oculta

# añadir la capa segunda capa oculta
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# añadir la capa de salida
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# tiene un solo nodo porque es una clasificacion binaria

# Compilar la RNA
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Ajustar el clasificador en el Conjunto de Entrenamiento
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
# epochs = nro de iteraciones global sobre todo el algoritmo

# Parte 3 - Evaluar el modelo y calcular predicciones finales

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = (y_pred > 0.5)
print('>0.5')
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('accuracy_score:', accuracy_score(y_test, y_pred) )

# ajuste de hiperparametros, tecnicas de validacion probablemente aumenten la precision
