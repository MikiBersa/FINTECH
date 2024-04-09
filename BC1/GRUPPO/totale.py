import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as clrs
import numpy as np

# Kmeans algorithm from scikit-learn
# PER IL MACHINE LEARNING
from sklearn.cluster import KMeans # K MEANS ALGORITMO DI APPLICAZIONE CLUSTERING
from sklearn.metrics import silhouette_samples, silhouette_score #  UTILIZZO DISTANZA SILHOUTTE
from sklearn.manifold import Isomap

DATA_FOLDER = './'
df_original = pd.read_excel(os.path.join(DATA_FOLDER, 'BankClients.xlsx'))

df_new  = df_original.iloc[:200, 1:]
# SEE THE STRUCTURE OF THE  DATA
print("Size of the dataset (row, col): ", df_new.shape) # DEFINIZIONE DELLA STRUTTURA
print("\nFirst 5 rows\n", df_new.head(n=5)) #

# CREAZIONE DELLE VARIABILI DUMMY E SCALE DEI NUMERI

# elaborazione dei dati per identificare le categorie
def is_categorical_column(column):
    # Conta i valori univoci nella colonna
    unique_values = np.unique(column)
    num_unique_values = len(unique_values)

    # print(unique_values, num_unique_values, len(column))
    # print(unique_values, num_unique_values)

    # Calcola la proporzione di valori univoci rispetto alla lunghezza totale della colonna
    unique_ratio = num_unique_values / len(column)

    # Se la proporzione è inferiore a una soglia arbitraria (ad esempio, 0.05),
    # considera la colonna come una variabile categorica
    # if unique_ratio < 0.005:
    if unique_ratio < 0.05:
        return True
    else:
        return False

typology = [is_categorical_column(df_new[colonna]) for colonna in df_new.columns]

# scalare i valori che sono numerico:
num = []
cat = []
values = df_new.values

print(typology)

for pos in range(len(df_new.columns.tolist())):
    if(typology[pos]):
       cat.append(values[: , pos])
    else:
       num.append(values[: , pos])

num = np.transpose(np.array(num))
# per i numeri faccio lo scalo in base alla media
# num = (num - num.mean()) / num.std()
minimo = np.min(df_new.values[:, 0])
maximo = np.max(df_new.values[:, 0])
print(df_new.shape[0])
for riga in range(df_new.shape[0]):
    num[riga, 0] = (df_new.values[riga, 0] - minimo) / (maximo - minimo )

cat = np.transpose(np.array(cat))


print(num.shape)
# print("NUM: ",num)
print(cat.shape)

# creazione dei dummyvar
cat_dummies = []
cat_i = pd.get_dummies(cat[:, 0])
cat_i = cat_i.values[:, :-1]
cat_dummies = cat_i

for col in range(1, cat.shape[1]):
    cat_i = pd.get_dummies(cat[:, col])
    cat_i = cat_i.values[:, :-1]
    cat_dummies = np.concatenate((cat_dummies, cat_i), axis=1)

X = np.concatenate((cat_dummies, num), axis=1)


print("X: ", X[0])

# DEFINIZIONE DELLA FUNZIONE DISTANZA:

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

def mixDistance(x, y):

    n_cat = 0
    n_num = 0

    # print("X :", x)
    # print("Y: ",y)

    # identificare dimensione dei categorie
    for colonne in range(len(X[0])):
        unique = np.unique(X[:, colonne])
        if len(unique) == 2 and unique[0] == 0 and unique[1] == 1:
            n_cat += 1

    # identidicare dimensione dei numeri per differenza
    n_num = len(X[0]) - n_cat

    NEW = np.vstack((x, y))
    # print(NEW)
    DCat = pdist(NEW[:, :n_cat], 'hamming')[0] # dimensione 1 x 1
    # print(DCat.shape)
    DNum = pdist(NEW[:, n_cat:], 'cityblock')[0] # dimensione 1 x 1
    # print(DNum.shape)

    weightC = n_cat / (n_num + n_cat)

    valore = weightC*DCat + (1 - weightC)*DNum
    # print(valore)
    return valore


# D = mixDistance(X[0, :], X[1, :])

# print(D)

# CLUSTERING CON IL KMEOIDS
from sklearn_extra.cluster import KMedoids

k = 3  # Number of clusters
n_replicates = 5

# Implement your custom distance calculation here
# This function should take two data points x1 and x2 as input
# and return the distance between them


kmedoids = KMedoids(n_clusters=k, metric=mixDistance)

# Fit the model to your data
kmedoids.fit(X)

# Get the cluster assignments
cluster_labels = kmedoids.labels_

# Get the cluster centroids (medoids)
cluster_medoids = kmedoids.cluster_centers_

print("LABEL: ", cluster_labels)
print("cluster_medoids: ", cluster_medoids)
print("Insertia: ", kmedoids.inertia_)


# IDENTIFICAZIONE INERZA
Ks = range(1, 6)
# KMeans(i) RUN THE ALGORITM WITH i CLUSTERS
# FALLO SUL DATA SET X
# POI CALCOLA  L'INERTIA
inertia = [KMedoids(n_clusters=i, metric=mixDistance).fit(X).inertia_ for i in Ks]


fig = plt.figure()
plt.plot(Ks, inertia, '-bo')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia (within-cluster sum of squares)')
plt.show()

range_n_clusters=[2,3,4,5,6,7,8,9,10]
for n_clusters in range_n_clusters:
    clusterer=KMedoids(n_clusters=n_clusters, metric=mixDistance)
    # COSì OTTENGO I LABEL
    cluster_labels=clusterer.fit_predict(X)
    # compio la silhouette everage score
    silhouette_avg=silhouette_score(X,cluster_labels)
    print("For n_clusters=", n_clusters,
          "The average silhouette_score is :", silhouette_avg)


from sklearn.manifold import TSNE

n_samples = X.shape[0]  # Numero di campioni nel dataset
D_custom = np.zeros((n_samples, n_samples))  # Matrice di distanza personalizzata

for i in range(n_samples):
    for j in range(i + 1, n_samples):
        D_custom[i, j] = mixDistance(X[i], X[j])
        D_custom[j, i] = D_custom[i, j]

# Creiamo un'istanza di TSNE e utilizziamo la matrice di distanza personalizzata come input
tsne = TSNE(metric='precomputed', init = "random", n_components=3, random_state=40)

# Calcoliamo la proiezione t-SNE utilizzando la matrice di distanza personalizzata come input
X_embedded = tsne.fit_transform(D_custom)

cluster_labels = kmedoids.labels_

# Visualizziamo i risultati in un grafico 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=cluster_labels, cmap='viridis')
ax.set_title('t-SNE Visualization in 3D with Custom Distance Metric')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.set_zlabel('t-SNE Dimension 3')
plt.show()