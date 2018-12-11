from scipy.io import arff
from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#---------------------LEITURA DOS DADOS---------------------

# Carrega o .arff
data1 = arff.loadarff(
		'/home/messias/Documentos/larvae/features/inception_resnet_v2.arff')
#data2 = arff.loadarff(
#		'/home/messias/Documentos/larvae/features/inception_v3.arff')
#data3 = arff.loadarff(
#		'/home/messias/Documentos/larvae/features/nasnet_large.arff')
#data4 = arff.loadarff(
#		'/home/messias/Documentos/larvae/features/resnet_50.arff')

# Transforma somente os dados em Data Frame
df1 = pd.DataFrame(data1[0])
#df2 = pd.DataFrame(data2[0])
#df3 = pd.DataFrame(data3[0])
#df4 = pd.DataFrame(data4[0])

# Imprime o Data Frame
# print(df4.head(n=9999))

# Armazenar as caracteristicas sem a Classe na variavel X
X1 = df1.iloc[:, 0:1536].values
#X2 = df2.iloc[:, 0:2048].values
#X3 = df3.iloc[:, 0:4032].values
#X4 = df4.iloc[:, 0:2048].values

#---------------------CLUSTERIZACAO COM KMEANS---------------------

# Inicializar o KMeans com 2 centroides
kmeans1 = KMeans(n_clusters=2, init='random')
#kmeans2 = KMeans(n_clusters=2, init='random')
#kmeans3 = KMeans(n_clusters=2, init='random')
#kmeans4 = KMeans(n_clusters=2, init='random')

# Executar passando como parametro os dados X
kmeans1.fit(X1)
#kmeans2.fit(X2)
#kmeans3.fit(X3)
#kmeans4.fit(X4)

# Verificar os centroides gerados pelo metodo
centers1 = kmeans1.cluster_centers_
#centers2 = kmeans2.cluster_centers_
#centers3 = kmeans3.cluster_centers_
#centers4 = kmeans4.cluster_centers_
#print(centers1[1])

# Variavel distance recebe uma tabela de distancia de todas amostras para
# os clusters
distance1 = kmeans1.fit_transform(X1)
#distance2 = kmeans2.fit_transform(X2)
#distance3 = kmeans3.fit_transform(X3)
#distance4 = kmeans4.fit_transform(X4)
print(distance1)
closests = distance1.flatten()
# Divide em par e impar para separar as distancias dos dois clusters
cluster1 = closests[0:][::2]
cluster2 = closests[1:][::2]
print(cluster1.shape)
print(cluster2.shape)
# Remove as amostras mais proximas e pega as mais longes agr
cluster1 = np.delete(cluster1, 565)
cluster2 = np.delete(cluster2, 276)
print(cluster1.shape)
print(cluster2.shape)
# Pega as menores distancias de cada amostra a cada cluster
c_1 = cluster1.argmin()
c_2 = cluster2.argmin()
print(c_1, c_2)


# Variavel labels recebe os clusters atribuidos a cada amostra
labels1 = kmeans1.labels_
#print(labels1)
#labels2 = kmeans2.labels_
#labels3 = kmeans3.labels_
#labels4 = kmeans4.labels_
#print(labels1)

#---------------------GRAFICOS KMEANS---------------------

# Grafico para Inception ResNet V2
#plt.figure(1)
#plt.scatter(X1[:, 0], X1[:, 1], s=100, c=labels1)
#plt.scatter(centers1[:, 0], centers1[:, 1], s=300, c='red', label='Centroids')
#plt.title('KMEANS: Inception ResNet V2')

# Grafico para Inception V3
#plt.figure(2)
#plt.scatter(X2[:, 0], X2[:, 1], s=100, c=labels2)
#plt.scatter(centers2[:, 0], centers2[:, 1], s=300, c='red', label='Centroids')
#plt.title('KMEANS: Inception V3')

# Grafico para NASNet Large
#plt.figure(3)
#plt.scatter(X3[:, 0], X3[:, 1], s=100, c=labels3)
#plt.scatter(centers3[:, 0], centers3[:, 1], s=300, c='red', label='Centroids')
#plt.title('KMEANS: NASNet Large')

# Grafico para ResNet 50
#plt.figure(4)
#plt.scatter(X4[:, 0], X4[:, 1], s=10, c=labels4)
#plt.scatter(centers4[:, 0], centers4[:, 1], s=30, c='red', label='Centroids')
#plt.title('KMEANS: ResNet 50')

#plt.show()

#---------------------CLUSTERIZACAO COM DBSCAN---------------------

# Inicializa o DBSCAN
#db1 = DBSCAN(eps=10, min_samples=10).fit(X1)
#db2 = DBSCAN(eps=10, min_samples=10).fit(X2)
#db3 = DBSCAN(eps=10, min_samples=10).fit(X3)
#db4 = DBSCAN(eps=10, min_samples=10).fit(X4)

#core_samples_mask1 = np.zeros_like(db1.labels_, dtype=bool)
#core_samples_mask1[db1.core_sample_indices_] = True
#core_samples_mask2 = np.zeros_like(db2.labels_, dtype=bool)
#core_samples_mask2[db2.core_sample_indices_] = True
#core_samples_mask3 = np.zeros_like(db3.labels_, dtype=bool)
#core_samples_mask3[db3.core_sample_indices_] = True
#core_samples_mask4 = np.zeros_like(db4.labels_, dtype=bool)
#core_samples_mask4[db4.core_sample_indices_] = True

# Recebendo os rotulos
#labels1 = db1.labels_
#labels2 = db1.labels_
#labels3 = db1.labels_
#labels4 = db1.labels_

# Numero de clusters nos labels ignorando os ruidos
#n_clusters_1 = len(set(labels1)) - (1 if -1 in labels1 else 0)
#n_clusters_2 = len(set(labels2)) - (1 if -1 in labels2 else 0)
#n_clusters_3 = len(set(labels3)) - (1 if -1 in labels3 else 0)
#n_clusters_4 = len(set(labels4)) - (1 if -1 in labels4 else 0)

#unique_labels1 = set(labels1)
#colors = [plt.cm.Spectral(each)
#          for each in np.linspace(0, 1, len(unique_labels1))]
#for k, col in zip(unique_labels1, colors):
#    if k == -1:
#        # Black used for noise.
#        col = [0, 0, 0, 1]

#    class_member_mask1 = (labels1 == k)
#    plt.figure(1)
#    xy = X1[class_member_mask1 & core_samples_mask1]
#    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#             markeredgecolor='k', markersize=14)
#
#    xy = X1[class_member_mask1 & ~core_samples_mask1]
#    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#             markeredgecolor='k', markersize=6)
#
#plt.title('Numero de clusters: %d' % n_clusters_1)
#
#unique_labels2 = set(labels2)
#colors = [plt.cm.Spectral(each)
#          for each in np.linspace(0, 1, len(unique_labels2))]
#for k, col in zip(unique_labels2, colors):
#    if k == -1:
#        # Black used for noise.
#        col = [0, 0, 0, 1]
#
#    class_member_mask2 = (labels2 == k)
#    plt.figure(2)
#    xy = X2[class_member_mask2 & core_samples_mask2]
#    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#             markeredgecolor='k', markersize=14)
#
#    xy = X2[class_member_mask2 & ~core_samples_mask2]
#    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#             markeredgecolor='k', markersize=6)
#
#plt.title('Numero de clusters: %d' % n_clusters_2)
#
#unique_labels3 = set(labels3)
#colors = [plt.cm.Spectral(each)
#          for each in np.linspace(0, 1, len(unique_labels3))]
#for k, col in zip(unique_labels3, colors):
#    if k == -1:
#        # Black used for noise.
#        col = [0, 0, 0, 1]
#
#    class_member_mask3 = (labels3 == k)
#    plt.figure(3)
#    xy = X3[class_member_mask3 & core_samples_mask3]
#    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#             markeredgecolor='k', markersize=14)
#
#    xy = X3[class_member_mask3 & ~core_samples_mask3]
#    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#             markeredgecolor='k', markersize=6)
#
#plt.title('Numero de clusters: %d' % n_clusters_3)

#unique_labels4 = set(labels4)
#colors = [plt.cm.Spectral(each)
#          for each in np.linspace(0, 1, len(unique_labels4))]
#for k, col in zip(unique_labels4, colors):
#    if k == -1:
#        # Black used for noise.
#        col = [0, 0, 0, 1]
#
#    class_member_mask4 = (labels4 == k)
#    plt.figure(4)
#    xy = X4[class_member_mask4 & core_samples_mask4]
#    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#             markeredgecolor='k', markersize=14)
#
#    xy = X4[class_member_mask4 & ~core_samples_mask4]
#    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#             markeredgecolor='k', markersize=6)
#
#plt.title('Numero de clusters: %d' % n_clusters_4)
#
#plt.show()
