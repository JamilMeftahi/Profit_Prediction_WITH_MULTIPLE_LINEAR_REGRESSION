# Régression Linéaire Multiple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Importer le dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Gérer les variables catégoriques

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        
         OneHotEncoder(), 
         [3]              
         )
    ],
    remainder='passthrough' 
)

X = transformer.fit_transform(X)

# Diviser le dataset entre le Training set et le Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Construction du modèle
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
  
print(regressor.predict(np.array([[1,0,0, 130000, 140000, 300000]])))



regressor.coef_

regressor.intercept_


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmse

# #ici on va essayer de trouver un modele mais avec moins de variables
# from sklearn.feature_selection import RFE

# rfe_5 = RFE(regressor, n_features_to_select=5)
# rfe_5.fit(X_train, y_train)
# y_pred = rfe_5.predict(X_test)
# print(r2_score(y_test, y_pred))

# for i in range(X_train.shape[1]):
# 	print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe_5.support_[i], rfe_5.ranking_[i]))

# rfe_4 = RFE(regressor, n_features_to_select=4)
# rfe_4.fit(X_train, y_train)
# y_pred = rfe_4.predict(X_test)
# print(r2_score(y_test, y_pred))

# for i in range(X_train.shape[1]):
# 	print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe_4.support_[i], rfe_4.ranking_[i]))

 

# Sauvegarder le modèle
pickle.dump(regressor, open('model.pkl', 'wb'))

    




