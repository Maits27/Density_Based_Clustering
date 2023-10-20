import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from main import DensityAlgorithm
import numpy as np
from loadSaveData import loadEmbeddings
import csv


def heuristicoEpsilonDBSCAN(nInstances, dimension):
    # Load vectors
    embeddingVectors = loadEmbeddings(length=nInstances, dimension=dimension)

    # Get distances using k-NN
    nn = NearestNeighbors(n_neighbors=dimension*2)
    nn.fit(embeddingVectors)
    distances, idx = nn.kneighbors(embeddingVectors)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]

    # Plot
    plt.plot(distances)
    plt.title(f'Eps heuristic for: {nInstances} instances, {dimension} dimensions')
    plt.xlabel('Instances')
    plt.ylabel('Distances')
    plt.yticks([i for i in range(0, int(distances.max())) if i % 2 == 0])
    plt.grid(True)
    plt.savefig(f'../img/elbowMethod/eps{nInstances}dim{dimension}')
    plt.show()


def barridoDBSCAN(nInstances, dimension, espilonList, minPtsList):
    # Load vectors
    embeddingVectors = loadEmbeddings(length=nInstances, dimension=dimension)

    # Barrido
    for eps in espilonList:
        for minPts in minPtsList:
            dbscan = DensityAlgorithm(vectors=embeddingVectors, epsilon=eps, minPt=minPts)
            dbscan.ejectuarAlgoritmo()
            dbscan.imprimir()
            silhouette = silhouette_score(embeddingVectors, dbscan.clusters)
            print(silhouette)
            saveInExcel(nInstances, dimension, eps, minPts, 3929, silhouette)


def barridoDoc2Vec(dimensionList):
    pass


def saveInExcel(nInstances, dimension, espilon, minPts, nClusters, silhouette):
    with open('../Barridos.csv', 'a') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow([nInstances, dimension, espilon, minPts, nClusters, silhouette])



barridoDBSCAN(nInstances=100, dimension=150, espilonList=[16], minPtsList=[150*2])
#heuristicoEpsilonDBSCAN(nInstances=20000, dimension=150)