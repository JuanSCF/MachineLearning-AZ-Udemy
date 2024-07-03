#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

# Udemy random forest regression

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Escalado de variables
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# sc_y = StandardScaler()
# X = sc_X.fit_transform(X)
# y = sc_y.fit_transform(y.reshape(-1, 1))

# Ajustar el random forest con el dataset
from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor(n_estimators=1000, random_state = 0)
regression.fit(X, y)


# Predicción de nuestros modelos con random forest
# fit_transform es para hacer el modelo
# transform es cuando ya tengo el modelo
y_pred = regression.predict( [[6.5]] )


# Visualización de los resultados del random forest
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.plot(X_grid, regression.predict(X_grid), color = "blue")
# plt.plot(X, regression.predict(X), color = "blue")
plt.scatter(X, y, color = "red")
plt.title("Modelo de Regresión")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()


