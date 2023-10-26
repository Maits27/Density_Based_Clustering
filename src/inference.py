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
from sklearn.decomposition import PCA
def create_embeddings():

    # GENERATE TEST.CSV --> reduceDataset.py

    #TRANSFORMERS TRAIN
    train=loadEmbeddings(length=10000,dimension=768,type='bert')

    #TRANSFORMERS TEST
    path = '../Datasets/Suicide_Detection100_test.csv'
    vectorizationMode = vectorization.bertTransformer # doc2vec, tfidf, bertTransformer
    rawData = loadRAW(path)
    test = vectorizationMode(rawData)
    return train,test

def make_clustering(data):
    # Clustering
    clusteringAlgorithm = clustering.DensityAlgorithmUrruela # DensityAlgorithmUrruela, DensityAlgorithm, DensityAlgorithm2, DBScanOriginal
    epsilon = 0.05
    minPts = 3
    algoritmo = clusteringAlgorithm(data, epsilon=epsilon, minPt=minPts,dim=768)
    algoritmo.ejecutarAlgoritmo()
    algoritmo.imprimir()
    clusters = algoritmo.clusters

    return clusters

def load_embeddings():
    train = loadEmbeddings(length=10000, dimension=768, type='bert')
    test = loadEmbeddings(length=100, dimension=768, type='bert')
    return train,test

def visualizar_instancias_por_clase():

    x_test=pd.read_csv('../Datasets/Suicide_Detection100_test.csv')
    class_counts = x_test['class'].value_counts()
    print(class_counts)

    # Crear un gráfico de barras
    plt.figure(figsize=(10, 6))  # Ajusta el tamaño del gráfico
    class_counts.plot(kind='bar')
    plt.title('Número de Instancias por Clase')
    plt.xlabel('Clase')
    plt.ylabel('Número de Instancias')
    plt.show()


def buscar_instancias_cluster(train,clusters, clusterNum):
    for i in range(len(train)):
        if clusters[i] == clusterNum:
            print(i)


def add_instances_to_test (train,test,instances):
    for i in range(len(train)):
        for instance in instances:
            if i==instance:
                test= np.vstack((test, train[i]))
    return test

def asignar_cluster_test_instancia(train,test,clusters):
    clusters_test=[]
    for test_instance in test:
            distancias=1- cosine_similarity(test_instance.reshape(1,-1),train)[0]
            instanciaCercana = np.argmin(distancias)
            cluster_asignado=clusters[instanciaCercana]
            clusters_test.append(cluster_asignado)
    return clusters_test

def asignar_cluster_test_centroide(train,test,clusters):
    clusters_test = []
    centroides = []
    for label in set(clusters):
        if label != -1:
            instanciasCluster = train[clusters == label]
            centroide = instanciasCluster.mean(axis=0)
            centroides.append(centroide)

    for test_instance in test:
        distancias = 1 - cosine_similarity(test_instance.reshape(1, -1), centroides)[0]
        cluster_asignado = np.argmin(distancias)
        clusters_test.append(cluster_asignado)
    return clusters_test,centroides


def reducir_dim(train, test,centroides):
    print('Dim train originally: ', np.array(train).shape)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(train)
    train_reducido = pca.transform(train)
    print('Dim train after PCA: ', train_reducido.shape)

    print('Dim test originally: ', np.array(test).shape)
    test_reducido = pca.transform(test)
    print('Dim test after PCA: ', test_reducido.shape)

    print('Dim centroides originally: ', np.array(centroides).shape)
    centroides_reducidos = pca.transform(centroides)
    print('Dim centroides after PCA: ', centroides_reducidos.shape)

    return train_reducido, test_reducido,centroides_reducidos


def grafico_instancia(train_reducido, clusters, test_reducido, clusters_test):
    colores = ['g', 'r', 'purple', 'y', 'k', 'orange', 'pink', 'brown', 'teal', 'lime', 'navy', 'gray']
    plt.figure(figsize=(16, 14))

    # TRAIN
    unique_labels = set(clusters) - {-1}

    # ruido
    noise_points = np.array([train_reducido[i] for i in range(len(train_reducido)) if clusters[i] == -1])
    plt.scatter(noise_points[:, 0], noise_points[:, 1], c='blue', label='Noise')

    # instancias train cluster
    for label in unique_labels:
        cluster_points = np.array([train_reducido[i] for i in range(len(train_reducido)) if clusters[i] == label])
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colores[label % len(colores)],
                    label=f'Cluster {label}')

    # TEST
    colores = ['lime', 'pink', 'orange', 'purple', 'pink', 'brown', 'teal', 'lime', 'navy', 'gray']
    unique_labels = set(clusters_test) - {-1}

    # ruido
    noise_points = np.array([test_reducido[i] for i in range(len(test_reducido)) if clusters_test[i] == -1])
    plt.scatter(noise_points[:, 0], noise_points[:, 1], c='c', label='Noise Test')

    # instancias test cluster
    for label in unique_labels:
        cluster_points = np.array([test_reducido[i] for i in range(len(test_reducido)) if clusters_test[i] == label])
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colores[label % len(colores)], label=f'Cluster {label} test')

    plt.title('Gráfico de Densidad basado en DBSCAN')
    plt.xlabel('Dimensión X')
    plt.ylabel('Dimensión Y')
    plt.legend()
    plt.show()


def grafico_centroide(train_reducido, clusters, test_reducido, clusters_test,centroides_reducidos):
    colores = ['g', 'r', 'purple', 'y', 'k', 'orange', 'pink', 'brown', 'teal', 'lime', 'navy', 'gray']
    plt.figure(figsize=(16, 14))

    # TRAIN
    unique_labels = set(clusters) - {-1}

    # ruido
    noise_points = np.array([train_reducido[i] for i in range(len(train_reducido)) if clusters[i] == -1])
    plt.scatter(noise_points[:, 0], noise_points[:, 1], c='blue', label='Noise')

    # instancias train cluster
    for label in unique_labels:
        cluster_points = np.array([train_reducido[i] for i in range(len(train_reducido)) if clusters[i] == label])
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colores[label % len(colores)],
                    label=f'Cluster {label}')

    # TEST
    colores = ['lime', 'pink', 'orange', 'purple', 'pink', 'brown', 'teal', 'lime', 'navy', 'gray']
    unique_labels = set(clusters_test) - {-1}

    # ruido
    noise_points = np.array([test_reducido[i] for i in range(len(test_reducido)) if clusters_test[i] == -1])
    plt.scatter(noise_points[:, 0], noise_points[:, 1], c='c', label='Noise Test')

    # instancias test cluster
    for label in unique_labels:
        cluster_points = np.array([test_reducido[i] for i in range(len(test_reducido)) if clusters_test[i] == label])
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colores[label % len(colores)], label=f'Cluster {label} test')

    #centroides
    centroides_points = np.array([centroides_reducidos[i] for i in range(len(centroides_reducidos))])
    plt.scatter(centroides_points[:, 0], centroides_points[:, 1], c='gray', marker='x', s=400, label='Centroide')

    plt.title('Gráfico de Densidad basado en DBSCAN')
    plt.xlabel('Dimensión X')
    plt.ylabel('Dimensión Y')
    plt.legend()
    plt.show()