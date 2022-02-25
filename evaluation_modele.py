"""
jeu de données contient des informations physico-chimiques de vins portugais ainsi que leur qualité telle que notée par des humains
prédire automatiquement la qualité sur la base de ces informations afin d’améliorer la production de vin, et de cibler le goût des consommateurs sur des marchés de niche
"""


########## BIBLIOTHÈQUES ##########

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection     # pour sampler nos données

# standardiser le training set, appliquer la même transformation au testing set
from sklearn import preprocessing

from sklearn import neighbors, metrics, dummy



########## PARCOURIR LES DONNÉES ##########

data = pd.read_csv("/Users/taoufiq/Documents/machine learning/evaluation d'un modele de prediction de qualite de vin/winequality-white.csv", sep=";")

print(data.head())

""" les données contiennent 12 colonnes, 11 correspondant à des indicateurs
physico-chimiques et 1 qui est la qualité du vin """

data.describe()

data.info()

data.isna().sum() # pas de valeur na

########## AFFICHER HISTOGRAMME POUR CHAQUE VARIABLE ##########

X = data[data.columns[:-1]].values  # toutes les colonnes sauf la dernière
y = data['quality'].values # on récupère les valeurs de la colonne qualité


fig = plt.figure(figsize=(16, 12))  # on définit la taille de notre figure


for i in range(X.shape[1]): # on boucle sur les 10 indicateurs physico-chimiques
    ax = fig.add_subplot(3,4, (i+1))
    h = ax.hist(X[:, i], bins=50, color='steelblue', density=True, edgecolor='none')
    ax.set_title(data.columns[i], fontsize=14)

plt.show()

""" les variables prennent des valeurs dans des ensembles différents
Il faut standardiser les données pour éviter que certaines variables ne domine
pas complètement les autres """


########## ENTRAÎNEMENT ##########

""" d'abord transformer ce problème en un problème de classification ->
séparer les bons vins des vins médiocres """

# on sépare y en 2 groupes, 0 pour les qualités < 6 et 1 sinon
y_class = np.where(y<6, 0, 1)

# Sampling, 30% des données dans le jeu de test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_class,
                                                                  test_size=0.3)


# standardiser le training set, appliquer la même transformation au testing set
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)


""" visualiser de nouveau les données pour vérifier que les variables prennent
des valeurs ayant des ordres de grandeur similaire """

fig = plt.figure(figsize=(16, 12))

for i in range(X_train_std.shape[1]):
    ax = fig.add_subplot(3,4, (i+1))
    h = ax.hist(X_train_std[:, i], bins=50, color = 'steelblue', density=True, edgecolor='none')
    ax.set_title(data.columns[i], fontsize=14)

plt.show()

########## SELECTION DE MODÈLE / VALIDATION CROISÉE ##########


""" méthode GridSearchCV pour faire validation croisée du paramètre k du KNN
sur le training set

Pour sélectionner un modèle, on compare les performances en validation croisée
sur un jeu d’entraînement"""

# Fixer les valeurs des hyperparamètres à tester
param_grid = {'n_neighbors' : [3, 5, 7, 9, 11, 13, 15]}

# Choisir un score à optimiser, 'accuracy' = proportion de prédictions correctes
score = 'accuracy'

# Créer classifieur KNN avec recherche d'hyperparamètre par validation croisée
clf = model_selection.GridSearchCV(
    neighbors.KNeighborsClassifier(), # un classifieur kNN
    param_grid,     # hyperparamètres à tester
    cv=5,           # nombre de folds de validation croisée
    scoring=score   # score à optimiser
)

# Optimiser ce classifieur sur le jeu d'entraînement
clf.fit(X_train_std, y_train)

# Afficher hyperparamètres optimaux  k=15
print("\n Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement : ", end='')
print(clf.best_params_)

# Afficher les performances correspondantes
print("Résultats de la validation croisée :\n")
for mean, std, params in zip(
        clf.cv_results_['mean_test_score'], # score moyen
        clf.cv_results_['std_test_score'],  # écart-type du score
        clf.cv_results_['params']           # valeur de l'hyperparamètre
    ) :

    print("{} = {:.3f} (+/-{:.03f}) for {}".format(
        score,
        mean,
        std*2,
        params ) )


""" la meilleure performance est 0.765 et atteinte avec 11 voivins """

""" regarder la performance sur le jeu de test. GridSearchCV ré-entraîne
automatiquement le meilleur modèle sur l’intégralité du jeu d’entraînement """
y_pred = clf.predict(X_test_std)
print("\n Sur le jeu de test :{:.3f}".format(
                                        metrics.accuracy_score(y_test,y_pred)))



########## MATRICE DE CONFUSION ########## y_true

# metrics.confusion_matrix(y_pred, y_pred)

########## COURBE AUROC ##########

# Traçons la courbe ROC du kNN
y_pred_proba = clf.predict_proba(X_test_std)[:, 1]
[fpr, tpr, thr] = metrics.roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='coral', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - specificite', fontsize=14)
plt.ylabel('Sensibilite', fontsize=14)

plt.show()

# AUROC
print(metrics.auc(fpr, tpr))

""" choisir un seuil de décision à partir de cette courbe -> fixe soit la spécificité, soit la sensibilité désirée et on cherche le seuil correspondant

si algorithme doit détecter efficacement les vins de mauvaise qualité, qui ne seront pas ensuite examinés par un expert humain. On veut alors limiter le nombre de faux négatifs, pour limiter le nombre de rejets infondés.
Fixons un taux de faux négatifs tolérable (proportion de positifs incorrectement prédits négatifs) de 5% -> sensibilité de 95% """

# FNR = FN/#pos = FN/(TP+FN) = 1 − TP/(TP+FN) = 1 − sensibilité

# indice du premier seuil pour lequel
# la sensibilité est supérieure à 0.95
idx = np.min(np.where(tpr > 0.95))

print("Sensibilité : {:.2f}".format(tpr[idx]))
print("Spécificité : {:.2f}".format(1-fpr[idx]))
print("Seuil : {:.2f}".format(thr[idx]))

""" un seuil de 0.29 garantit une sensibilité = 0.97 et une spécificité = 0.21
soit un taux de faux positifs = 79%."""


########## COMPARER MODELE KNN A DES APPROCHES NAÏVES ##########

# standardiser le training set, appliquer la même transformation au testing set
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)

# Entraînons un KNN avec k=11
knn = neighbors.KNeighborsRegressor(n_neighbors=11)
knn.fit(X_train_std, y_train)

# appliquons le KNN pour prédire les étiquettes du jeu de test
y_pred = knn.predict(X_test_std)

# Calculons la RMSE correspondante
print("RMSE : {:.2f}".format(np.sqrt( metrics.mean_squared_error(y_test, y_pred) )))

# visualiser les résultats, en abscisse les vraies valeurs des étiquettes et en
# ordonnée les valeurs prédites
plt.scatter(y_test, y_pred, color='coral')

plt.show()

# cercles dont taille proportionnelle au nombre de points présents à ces coord

sizes = {} # clé : coordonnées ; valeur : nombre de points à ces coordonnées
for (yt, yp) in zip(list(y_test), list(y_pred)):
    if (yt, yp) in sizes:
        sizes[(yt, yp)] += 1
    else:
        sizes[(yt, yp)] = 1

keys = sizes.keys()
plt.scatter(
        [k[0] for k in keys], # vraie valeur (abscisse)
        [k[1] for k in keys], # valeur predite (ordonnee)
        s=[sizes[k] for k in keys], # taille du marqueur
        color='coral', alpha =0.8)

plt.show()


# approche naïve consistant à prédire des valeurs aléatoires, distribuées uniformément entre les valeurs basse et haute des étiquettes du testing test
y_pred_random = np.random.randint(np.min(y), np.max(y), y_test.shape)

# Calculons RMSE correspondante
print("RMSE : {:.2f}".format(np.sqrt(metrics.mean_squared_error(y_test, y_pred_random))))


"""  RMSE supérieur à la RMSE obtenue par le kNN.
Notre modèle a ainsi réussi à mieux apprendre qu'un modèle aléatoire

beaucoup de nos vins ont une note de 6, et beaucoup de prédictions sont autour
de cette valeur. Comparons notre modèle à un modèle aléatoire retournant la
moyenne des étiquettes du testing test """


dum = dummy.DummyRegressor(strategy='mean')

# Entraînement
dum.fit(X_train_std, y_train)

# Prédiction sur le testing test
y_pred_dum = dum.predict(X_test_std)

# Evaluer
print("RMSE : {:.2f}".format(np.sqrt(metrics.mean_squared_error(y_test, y_pred_dum)) ))

"""  RMSE supérieur à la RMSE obtenue par le kNN.
Notre modèle a  a donc appris plus que la moyenne des étiquette """