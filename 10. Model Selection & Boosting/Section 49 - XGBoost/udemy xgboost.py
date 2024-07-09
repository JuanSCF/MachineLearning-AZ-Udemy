#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
# pip install xgboost==1.1.0

# XGBoost
# significant enhancement to the gradient boosting algorithm
# documentation: https://xgboost.readthedocs.io/en/stable/
# xgboost uses random forest and or ann to improve the gradient descent





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


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ajustar el set de entrenamiento con xgboost
from xgboost import XGBClassifier
classifier = XGBClassifier() # tipico valor para árboles = n_estimators es 300. max_depth = tamaño red neuronal
# usar grid search para optimizar learning_rate, n_estimators, por ej
classifier.fit(X_train, y_train)


# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred  = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
print('>0.5')
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('accuracy_score:', accuracy_score(y_test, y_pred) )


# Aplicar k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10) # el parametro n_jobs sirve para entrenar en paralelo los k modelos
print('Cross Validation')
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
