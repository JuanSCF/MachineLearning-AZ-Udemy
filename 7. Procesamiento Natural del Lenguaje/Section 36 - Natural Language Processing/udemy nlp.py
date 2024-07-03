#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Natural Language Processing (NLP)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Limpieza de texto
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = [] # coleccion de textos limpiados que puede ser usada para cualquier tipo de algoritmos

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Crear el Bag of Words, matriz dispersa
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # transforma una frase en vector. max features delimita el numero de palabras a las más frecuentes
X = cv.fit_transform(corpus).toarray()
# fit analiza/crea el modelo. transform aplica el modelo a los datos
# fit transform hace las dos cosas
y = dataset.iloc[:, 1].values


# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('accuracy_score:', accuracy_score(y_test, y_pred) )

# Calcular la precisión del modelo
accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
print(accuracy)
