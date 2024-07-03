#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# (Associative rule learning) Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import apyori

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
X = dataset.iloc[:, [3, 4]].values

# apriori expects a list of lists as input
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Entrenar el algoritmo de apriori 
from apyori import apriori
rules = apriori(transactions, min_support=3*7/7500, min_confidence=0.2, min_lift=3, min_length=2)

# Visualising the results
results = list(rules)

results[0]