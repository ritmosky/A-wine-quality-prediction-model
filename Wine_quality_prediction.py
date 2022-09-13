#!/usr/bin/env python
# coding: utf-8

# # A-wine-quality-prediction-model-evaluation

# ### Contexte
# 
# Ce dataset contient des informations physico-chimiques de vins portugais ainsi que leur qualité telle que notée par des humains.Le but de ce code est de prédire automatiquement la qualité du vin sur la base de ces informations afin d’améliorer la production de vin, et de cibler le goût des consommateurs sur des marchés de niche. Par la suite nous allons évaluer ce modèle et le comparer à différents algorithmes pour ainsi juger de la pertinence de notre modèle.

# ### Bibliothèques

# In[168]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing, neighbors, metrics, linear_model


# ### Data Exploration

# In[34]:


data = pd.read_csv("winequality-white.csv", sep=";")

data.head()


# In[35]:


data.shape


# In[36]:


data.info()


# In[37]:


data.describe()


# In[38]:


data.isna().sum()


# In[68]:


sns.heatmap(data.corr())


# In[164]:


y = data['quality']
X = data.drop('quality', axis=1)
print('\n', y.shape, X.shape)


# In[95]:


plt.figure(figsize=(16, 12))
for name, i in zip(data.columns, range(X_train_std.shape[1])): 
    plt.subplot(3,4,i+1)
    plt.hist(X.iloc[:, i], bins=50)
    plt.title(name)
plt.show()


# ### Data Preprocessing

# Normalisation 
# 
# MinMaxScaler -> placer entre 0 et 1 (X-Xmin)/(Xmax-Xmin) mais il faut enlever les outliers avant
# StandardScaler -> centrée et réduire (X-mean(X))/std(X) mais il faut enlever les outliers avant
# RobustScaler -> rend moins sensible aux outliers (X-mediane)/IQR
# 
# un_transformer.inverse_transform(np.array) -> repasser à l'échelle réelle

# In[116]:


# Pour séparer les vins en 2
y_class = np.where(y<6, 0, 1)

# Sampling
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_class,
                                                                  test_size=0.2)

# Transformer 
scaler = preprocessing.StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.fit_transform(X_test)

plt.figure(figsize=(16, 12))
for name, i in zip(data.columns, range(X_train_std.shape[1])): 
    plt.subplot(3,4,i+1)
    plt.hist(X_train_std[:, 1], bins=50)
    plt.title(name)
plt.show()

### Validation Curve (teste toutes les valeurs pour un hyperparamètre donné par Cross Validation)

k = np.arange(1,50)

train_score, val_score = model_selection.validation_curve(knn,
                                                         X_train_std, y_train,
                                                         'n_neighbors',
                                                         k, cv=5)

val_score.mean(axis=1)

plt.plot(k, val_score.mean(axis=1), label='validation')
plt.plot(k, train_score.mean(axis=1), label='train')
plt.ylabel('score')
plt.xlabel('n-neighbors')
plt.legend()
# ### Cross Validation & Data Modeling

# In[162]:


knn_score = model_selection.cross_val_score(neighbors.KNeighborsClassifier(), 
                                X_train_std, y_train, 
                                cv=5, 
                                scoring='accuracy')

sgdc_score = model_selection.cross_val_score(linear_model.SGDClassifier(), 
                                X_train_std, y_train, 
                                cv=5, 
                                scoring='accuracy')

print('\nKnn score:', knn_score.mean())
print('Sgdc score:', sgdc_score.mean())


# In[161]:


# Estimator
knn = neighbors.KNeighborsClassifier()
knn.fit(X_train_std, y_train)

print('\nTrain score:',knn.score(X_train_std, y_train))
print('Test score:',knn.score(X_test_std, y_test))


# ### GridSearchCV 

# In[140]:


param_grid = {'n_neighbors': np.arange(1,20),
              'metric': ['euclidean', 'manhattan']}

grid = model_selection.GridSearchCV(
    neighbors.KNeighborsClassifier(), 
    param_grid,     
    cv=5,           
    scoring='accuracy')

grid.fit(X_train_std, y_train)

print('\nGridSearchCV best score:', grid.best_score_)
print('GridSearchCV best params:', grid.best_params_)
print('GridSearchCV best estimators:', grid.best_estimator_)

best_knn = grid.best_estimator_
print('\n\nTest score sans GridSearchCV:', knn.score(X_test_std, y_test))
print('Test score avec GridSearchCV:', best_knn.score(X_test_std, y_test))


# ### Confusion Matrix (montrer les erreurs de prédiction)

# In[125]:


metrics.confusion_matrix(y_test, best_knn.predict(X_test_std))


# ### Metrics (analyse des erreurs)

# In[167]:


print('\nMAE:', metrics.mean_absolute_error(y_test, best_knn.predict(X_test_std)))

err = np.abs(y_test - best_knn.predict(X_test_std))
plt.hist(err, bins=50)
plt.show()


# ### Learning Curve (évolution des erformances en fonction de la quantité de données)

# In[137]:


N, train_score, val_score = model_selection.learning_curve(best_knn, 
                               X_train_std, y_train, 
                               train_sizes= np.linspace(0.2, 1, 10),
                               cv=5)

plt.plot(N, train_score.mean(axis=1), label='train')
plt.plot(N, val_score.mean(axis=1), label='validation')
plt.xlabel('train sizes')
plt.legend()
plt.show()

