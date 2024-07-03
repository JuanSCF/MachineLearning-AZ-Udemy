#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Upper Confidence Bound (UCB)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
# X = dataset.iloc[:, :].values


# Algoritmo de UCB
N = 10000
d = 10
number_of_selections = [0] * d
sum_of_rewards = [0] * d
ads_selected = []
total_reward = 0
for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range(0, d):
        if number_of_selections[i] > 0:
            average_reward = sum_of_rewards[i] / number_of_selections[i]
            delta_i = np.sqrt(3./2. * np.log(n+1)/number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] += reward
    total_reward += reward

    
# Visualising the results
# Histograma de resultados
plt.hist(ads_selected)
plt.title('Histograma de anuncios')
plt.xlabel('Anuncio')
plt.ylabel('Frecuencia')
plt.show()