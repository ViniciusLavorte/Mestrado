from scipy.io import arff
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt

#---------------------LEITURA DOS DADOS---------------------

#Carrega o .arff
data1 = arff.loadarff('/home/messias/Documentos/larvae/features/inception_resnet_v2.arff')
data2 = arff.loadarff('/home/messias/Documentos/larvae/features/inception_v3.arff')
data3 = arff.loadarff('/home/messias/Documentos/larvae/features/nasnet_large.arff')
data4 = arff.loadarff('/home/messias/Documentos/larvae/features/resnet_50.arff')

#Transforma somente os dados em Data Frame
df1 = pd.DataFrame(data1[0])
df2 = pd.DataFrame(data2[0])
df3 = pd.DataFrame(data3[0])
df4 = pd.DataFrame(data4[0])

#Imprime o Data Frame
#print(df4.head(n=9999))

#Armazenar as caracteristicas sem a Classe na variavel X
X1 = df1.iloc[:, 0:1536].values
X2 = df2.iloc[:, 0:2048].values
X3 = df3.iloc[:, 0:4032].values
X4 = df4.iloc[:, 0:2048].values

#Passar os rotulos para uma lista e substitui por numeros para poder aplicar os metodos (0 = impurity 1 = classe1)
labels1 = df1["class"].tolist()
aux = 0
for i in labels1:
    if i == 'impurity':
        labels1[aux] = 0
    else:
        labels1[aux] = 1
    aux = aux + 1
labels2 = df2["class"].tolist()
aux = 0
for i in labels2:
    if i == 'impurity':
        labels2[aux] = 0
    else:
        labels2[aux] = 1
    aux = aux + 1
labels3 = df3["class"].tolist()
aux = 0
for i in labels3:
    if i == 'impurity':
        labels3[aux] = 0
    else:
        labels3[aux] = 1
    aux = aux + 1
labels4 = df4["class"].tolist()
aux = 0
for i in labels4:
    if i == 'impurity':
        labels4[aux] = 0
    else:
        labels4[aux] = 1
    aux = aux + 1

#Aplica os metodos T-SNE e PCA
X1_tsne = TSNE().fit_transform(X1)
X1_pca = PCA().fit_transform(X1)
X2_tsne = TSNE().fit_transform(X2)
X2_pca = PCA().fit_transform(X2)
X3_tsne = TSNE().fit_transform(X3)
X3_pca = PCA().fit_transform(X3)
X4_tsne = TSNE().fit_transform(X4)
X4_pca = PCA().fit_transform(X4)

#Plotar os resultados
plt.figure(1)
plt.subplot(121)
plt.scatter(X1_tsne[:, 0], X1_tsne[:, 1], c=labels1)
plt.title('T-SNE Inception ResNet V2')
plt.subplot(122)
plt.scatter(X1_pca[:, 0], X1_pca[:, 1], c=labels1)
plt.title('PCA Inception ResNet V2')

plt.figure(2)
plt.subplot(121)
plt.scatter(X2_tsne[:, 0], X2_tsne[:, 1], c=labels2)
plt.title('T-SNE Inception V3')
plt.subplot(122)
plt.scatter(X2_pca[:, 0], X2_pca[:, 1], c=labels2)
plt.title('PCA Inception V3')

plt.figure(3)
plt.subplot(121)
plt.scatter(X3_tsne[:, 0], X3_tsne[:, 1], c=labels3)
plt.title('T-SNE NASNet Large')
plt.subplot(122)
plt.scatter(X3_pca[:, 0], X3_pca[:, 1], c=labels3)
plt.title('PCA NASNet Large')

plt.figure(4)
plt.subplot(121)
plt.scatter(X4_tsne[:, 0], X4_tsne[:, 1], c=labels4)
plt.title('T-SNE ResNet 50')
plt.subplot(122)
plt.scatter(X4_pca[:, 0], X4_pca[:, 1], c=labels4)
plt.title('PCA ResNet 50')

plt.show()