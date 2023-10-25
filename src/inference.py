import pandas as pd
import numpy as np
from loadSaveData import loadRAW
from tokenization import tokenize
import vectorization
import clustering
import evaluation
from loadSaveData import loadEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
def get_train_and_test():

    # GENERATE TEST.CSV --> reduceDataset.py

    #TRANSFORMERS TRAIN
    train=loadEmbeddings(length=10000,dimension=768,type='bert')

    #TRANSFORMERS TEST
    path = '../Datasets/Suicide_Detection100_test.csv'
    vectorizationMode = vectorization.bertTransformer # doc2vec, tfidf, bertTransformer
    rawData = loadRAW(path)
    test = vectorizationMode(rawData)
    return train,test

def dbscan_clustering(data):
    # Clustering
    clusteringAlgorithm = clustering.DBScanOriginal # DensityAlgorithmUrruela, DensityAlgorithm, DensityAlgorithm2, DBScanOriginal
    epsilon = 4
    minPts = 5
    algoritmo = clusteringAlgorithm(data, epsilon=epsilon, minPt=minPts)
    algoritmo.ejecutarAlgoritmo()
    algoritmo.imprimir()
    clusters = algoritmo.clusters

    return clusters
"""

x_test=pd.read_csv('../Datasets/Suicide_Detection100_test.csv')
print(x_test.head())
# Contar el número de instancias por clase
class_counts = x_test['class'].value_counts()

# Crear un gráfico de barras
plt.figure(figsize=(10, 6))  # Ajusta el tamaño del gráfico
class_counts.plot(kind='bar')

# Imprimir el número exacto de instancias por clase
print(class_counts)
plt.title('Número de Instancias por Clase')
plt.xlabel('Clase')
plt.ylabel('Número de Instancias')
plt.show()
"""

train,test=get_train_and_test()

# estas dos lineas para cuando ya tienes creados los embeddings
#train=loadEmbeddings(length=10000,dimension=768,type='bert')
#test=loadEmbeddings(length=100,dimension=768,type='bert')

clusters=dbscan_clustering(train)


"""
for i in range(len(train)):
    if clusters[i]==1:
        print(i)
# instancias del clsuter 1 --> 390, 936, 3777, 4084, 4510, 7968
"""
#añadimos al test por ejemplo esas dos instancias (para que tenga alguna de train)
test = np.vstack((test, train[4084]))
test = np.vstack((test, train[4510]))

#habria que buscar instancias ruido


for test_instance in test:
        distancias=1- cosine_similarity(test_instance.reshape(1,-1),train)[0]
        instanciaCercana = np.argmin(distancias)
        clusters_asignado=clusters[instanciaCercana]
        print("La instancia se ha asignado al cluster ",clusters_asignado)


#REDUCIR DIMESIONES
from sklearn.decomposition import PCA
print('Dim train originally: ',np.array(train).shape)
pca = PCA(n_components=2,random_state=42)
pca.fit(train)
train_reducido = pca.transform(train)
print('Dim train after PCA: ',train_reducido.shape)

print('Dim test originally: ',np.array(test).shape)
test_reducido = pca.transform(test)
print('Dim train after PCA: ',test_reducido.shape)


#GRAFICO
unique_labels = set(clusters) - {-1}

colores = ['g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink', 'brown', 'teal', 'lime', 'navy', 'gray']
plt.figure(figsize=(14, 12))

# ruido
noise_points = np.array([train_reducido[i] for i in range(len(train_reducido)) if clusters[i] == -1])
plt.scatter(noise_points[:, 0], noise_points[:, 1], c='blue', label='Noise')

# instancias train cluster
for label in unique_labels:
    cluster_points = np.array([train_reducido[i] for i in range(len(train_reducido)) if clusters[i] == label])
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colores[label % len(colores)], label=f'Cluster {label}')

# instancias test cluster
test_points=np.array([test_reducido[i] for i in range(len(test_reducido))])
plt.scatter(test_points[:, 0], test_points[:, 1], c='red', marker='x',s=200, label='Centroide')


plt.title('Gráfico de Densidad basado en DBSCAN')
plt.xlabel('Dimensión X')
plt.ylabel('Dimensión Y')
plt.legend()
plt.show()