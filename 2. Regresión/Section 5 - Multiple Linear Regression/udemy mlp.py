# Regresion lineal multiple

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# print(X)
# print(y)


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# print(X)

# Evitar la trampa de las variables ficticias (dummies)! multicolineality
X = X[:, 1:]

# Taking care of missing data
'''
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)
'''



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)


# Feature Scaling
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
print(X_test)
'''

# Ajustar el modelo de regresión lineal múltiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predicción de los resultados en el conjunto de testing
y_pred = regression.predict(X_test)

# plt.scatter(np.arange(len(y_test)), y_test, color = 'red')
# plt.plot(np.arange(len(y_test)), y_pred, color = 'blue')
# plt.show()

# construir el modelo óptimo de RLM utilizando la eliminación hacia atrás
import statsmodels.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
# axis = 0 añade fila. axis = 1 añade columna

SL = 0.05

# OLS = Ordinary Least Squares toma solo listas de listas, no arrays!
X_opt = X[:, [0, 1, 2, 3, 4, 5]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()

# ahora hay que eliminar la variable con el p-valor más alto

X_opt = X[:, [0, 1, 3, 4, 5]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3, 5]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()


# Automatización del proceso de eliminación hacia atrás
# usando solamente p-valores
# import statsmodels.formula.api as sm
def backwardElimination(x, sl):    
    numVars = len(x[0])    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

# usando p-valores y R^2
# import statsmodels.formula.api as sm
def backwardElimination(x, SL):    
    numVars = len(x[0])    
    temp = np.zeros((50,6)).astype(int)    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        adjR_before = regressor_OLS.rsquared_adj.astype(float)        
        if maxVar > SL:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    temp[:,j] = x[:, j]                    
                    x = np.delete(x, j, 1)                    
                    tmp_regressor = sm.OLS(y, x.tolist()).fit()                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
                    if (adjR_before >= adjR_after):                        
                        x_rollback = np.hstack((x, temp[:,[0,j]]))                        
                        x_rollback = np.delete(x_rollback, j, 1)     
                        print (regressor_OLS.summary())                        
                        return x_rollback                    
                    else:                        
                        continue    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)