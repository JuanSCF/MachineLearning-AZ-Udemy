#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:38:08 2019

@author: juan
"""

# Redes Neuronales Convolucionales


# Parte 1 - Construir el modelo de CNN
# from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Inicializar la CNN
cnn = Sequential()

# Paso 1 - Convolución
cnn.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Paso 2 - Max Pooling
# reduce el tamaño de la imagen/mapas de caracteristicas. reduciendo el numero de nodos y facilitando la convergencia del algoritmo
cnn.add(MaxPooling2D(pool_size = (2, 2)))

# segunda capa
cnn.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))

# tercera
cnn.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))

# Paso 3 - Flattening
cnn.add(Flatten())

# Paso 4 - Full Connection
cnn.add(Dense(units = 128, activation = 'relu'))
# una opcion comun es tomar el promedio de nodos entre la capa de entrada y la capa de salida
# en este caso no es muy buena idea porque la capa de entrada es muy grande
cnn.add(Dense(units = 128, activation = 'relu'))
cnn.add(Dense(units = 1, activation = 'sigmoid'))
# como es un sistema de clasificacion binario, la capa de salida basta con un solo nodo. se utiliza sigmoid para probabilidades


#para mejorar la precision de la red, se suele añadir más capas convolucionales + max pooling, o añadir capas ocultas del paso 4


# Compilar la CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Parte 2 - Ajustar la CNN a las imágenes para entrenar

# Preprocesado de las imágenes para evitar el overfitting. Le va aplicando transformaciones aleatorias a las imagenes de modo que todo lo que se aprende en el lote de informacion vaya modificandose de una iteracion a la siguiente. Sirve para enriquecer el dataset sin imagenes nuevas

# from keras.preprocessing.image import ImageDataGenerator
from keras.src.legacy.preprocessing.image import ImageDataGenerator
batch_size = 32

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
training_dataset = train_datagen.flow_from_directory(
        'dataset/training_set',  # this is the target directory
        target_size=(64, 64),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
test_dataset = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='binary')

# model.fit_generator is deprecated!
cnn.fit(
        training_dataset,
        steps_per_epoch=8000,# // batch_size, # aca poner el numero total de imagenes en el training set
        epochs=25,
        validation_data=test_dataset,
        validation_steps=2000)# // batch_size)

# cnn.save_weights('first_try.weights.h5')  # always save your weights after training or during training