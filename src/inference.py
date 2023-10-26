import statistics

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
def create_test_embeddings():
    #TRANSFORMERS TEST
    path = '../Datasets/Suicide_Detection_test2000(train10000).csv'
    vectorizationMode = vectorization.bertTransformer # doc2vec, tfidf, bertTransformer
    rawData = loadRAW(path)
    test = vectorizationMode(rawData)
    return test



def make_clustering(data):
    # Clustering
    clusteringAlgorithm = clustering.DBScanOriginal # DensityAlgorithmUrruela, DensityAlgorithm, DensityAlgorithm2, DBScanOriginal
    epsilon = 0.05
    minPts = 3
    algoritmo = clusteringAlgorithm(data, epsilon=epsilon, minPt=minPts)
    algoritmo.ejecutarAlgoritmo()
    algoritmo.imprimir()
    clusters = algoritmo.clusters

    return clusters


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

def asignar_cluster_test(train,test,clusters,):
    clusters_test=[]
    for test_instance in test:
            distancias=1- cosine_similarity(test_instance.reshape(1,-1),train)[0]
            instanciaCercana = np.argmin(distancias)
            cluster_asignado=clusters[instanciaCercana]
            clusters_test.append(cluster_asignado)
            """ COGER K instancias (falta el parametro)
            kInstanciasMasCercanas = np.argsort(distancias)[:kVecinos]
            clusters_asignados = [clusters[i] for i in kInstanciasMasCercanas]
            cluster_asignado = statistics.mode(clusters_asignados)
            clusters_test.append(cluster_asignado)
            """
    return clusters_test


def reducir_dim(train, test,dim):
    print('Dim train originally: ', np.array(train).shape)
    pca = PCA(n_components=dim, random_state=42)
    pca.fit(train)
    train_reducido = pca.transform(train)
    print('Dim train after PCA: ', train_reducido.shape)

    print('Dim test originally: ', np.array(test).shape)
    test_reducido = pca.transform(test)
    print('Dim test after PCA: ', test_reducido.shape)

    return train_reducido, test_reducido


def grafico(train_reducido, clusters, test_reducido, clusters_test):
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

def grafico_3d(train_reducido, clusters, test_reducido, clusters_test):
    colores = ['g', 'r', 'purple', 'y', 'k', 'orange', 'pink', 'brown', 'teal', 'lime', 'navy', 'gray']
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # TRAIN
    unique_labels = set(clusters) - {-1}

    # ruido
    noise_points = np.array([train_reducido[i] for i in range(len(train_reducido)) if clusters[i] == -1])
    ax.scatter(noise_points[:, 0], noise_points[:, 1],noise_points[:, 2], c='blue', label='Noise')

    # instancias train cluster
    for label in unique_labels:
        cluster_points = np.array([train_reducido[i] for i in range(len(train_reducido)) if clusters[i] == label])
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1],cluster_points[:, 2], c=colores[label % len(colores)],
                    label=f'Cluster {label}')

    # TEST
    colores = ['lime', 'pink', 'orange', 'purple', 'pink', 'brown', 'teal', 'lime', 'navy', 'gray']
    unique_labels = set(clusters_test) - {-1}

    # ruido
    noise_points = np.array([test_reducido[i] for i in range(len(test_reducido)) if clusters_test[i] == -1])
    ax.scatter(noise_points[:, 0], noise_points[:, 1],noise_points[:, 2], c='c', label='Noise Test')

    # instancias test cluster
    for label in unique_labels:
        cluster_points = np.array([test_reducido[i] for i in range(len(test_reducido)) if clusters_test[i] == label])
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1],cluster_points[:, 2], c=colores[label % len(colores)], label=f'Cluster {label} test')

    ax.set_xlabel('Dimensión X')
    ax.set_ylabel('Dimensión Y')
    ax.set_zlabel('Dimensión Z')
    plt.title('Gráfico de Densidad basado en DBSCAN (3D)')
    plt.legend()
    plt.show()


