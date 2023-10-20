import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from main import DensityAlgorithm, DBScanOriginal
import numpy as np
from loadSaveData import loadEmbeddings
import csv
import optuna


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
            dbscan = DBScanOriginal(vectors=embeddingVectors, epsilon=eps, minPt=minPts)
            dbscan.ejectuarAlgoritmo()
            numClusters = dbscan.getNumClusters()
            if numClusters != 1:
                silhouette = silhouette_score(embeddingVectors, dbscan.clusters)
            noiseInstances = dbscan.getNoiseInstances()
            saveInCSV(nInstances, dimension, eps, minPts, noiseInstances, silhouette)


def barridoDoc2Vec(dimensionList):
    pass


def saveInCSV(nInstances, dimension, espilon, minPts, nClusters, silhouette):
    with open('../Barridos.csv', 'a') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow([nInstances, dimension, espilon, minPts, nClusters, silhouette])


def objective(trial):    
    epsilon = trial.suggest_float('epsilon', 0.1, 10.0)
    minPt = trial.suggest_int('minPt', 2, 20)
    # Utiliza los valores sugeridos por Optuna para la ejecución    
    algoritmo = DBScanOriginal(documentVectors, epsilon=epsilon, minPt=minPt)
    algoritmo.ejecutarAlgoritmo()
    # Devuelve el número de instancias de ruido (puedes usar otra métrica)    
    return algoritmo.getNoiseInstances()


def barridoDBSCANOPtuna():
    study = optuna.create_study(direction='minimize')  # Optimiza para minimizar el ruido
    # Realiza la optimización de los parámetros
    study.optimize(objective, n_trials=100)
    # Obtiene los mejores parámetros encontrados
    best_epsilon = study.best_params['epsilon']
    best_minPt = study.best_params['minPt']


barridoDBSCAN(nInstances=100, dimension=150, espilonList=[16], minPtsList=[150*2])
#heuristicoEpsilonDBSCAN(nInstances=20000, dimension=150)