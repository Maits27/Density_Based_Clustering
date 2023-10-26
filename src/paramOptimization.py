import sys

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from clustering import DBScanOriginal, DensityAlgorithmUrruela
import numpy as np
from loadSaveData import loadEmbeddings, saveInCSV, saveInCSV2
import csv
import optuna
import plotly.express as px
import pandas as pd

optunaNCluster = 0


def heuristicoEpsilonDBSCAN(nInstances, dimension, type):
    # Load vectors
    embeddingVectors = loadEmbeddings(length=nInstances, dimension=dimension, type=type)

    # Get distances using k-NN
    nn = NearestNeighbors(n_neighbors=dimension * 2)
    nn.fit(embeddingVectors)
    distances, idx = nn.kneighbors(embeddingVectors)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]

    # Plot
    plt.plot(distances)
    plt.title(f'Eps heuristic for: {nInstances} instances, {dimension} dimensions')
    plt.xlabel('Instances')
    plt.ylabel('Distances')
    plt.yticks([i for i in range(0, int(distances.max())) if i % 2 == 0])
    plt.grid(True)
    plt.savefig(f'../img/elbowMethod/eps{nInstances}dim{dimension}')
    plt.show()


def barridoDBSCAN(nInstances, dimension, type, espilonList, minPtsList):
    # Load vectors
    embeddingVectors = loadEmbeddings(length=nInstances, dimension=dimension, type=type)

    # Barrido
    for eps in espilonList:
        for minPts in minPtsList:
            dbscan = DBScanOriginal(vectors=embeddingVectors, epsilon=eps, minPt=minPts)
            dbscan.ejecutarAlgoritmo()
            numClusters = dbscan.getNumClusters()
            print(numClusters)
            if numClusters > 1:
                print(dbscan.clusters)
                print(numClusters)
                silhouette = silhouette_score(embeddingVectors, dbscan.clusters)
            else:
                silhouette = 0
            noiseInstances = dbscan.getNoiseInstances()

            saveInCSV(nInstances=nInstances,
                      dimension=dimension,
                      espilon=eps,
                      minPts=minPts,
                      nClusters=numClusters,
                      silhouette=silhouette)


def barridoDoc2Vec(dimensionList):
    pass


def objective(trial, loadedEmbedding):
    epsilon = trial.suggest_float('epsilon', 0.005, 1.0, step=0.0001)
    minPt = trial.suggest_int('minPt', 5, 16)


    # Utiliza los valores sugeridos por Optuna para la ejecución
    algoritmo = DensityAlgorithmUrruela(loadedEmbedding, epsilon=epsilon, minPt=minPt)
    algoritmo.ejecutarAlgoritmo()

    optunaNCluster = algoritmo.getNumClusters()

    # TODO: Calcular media puntos cluster

    if algoritmo.getNumClusters() > 1:
        media_puntos_cluster = (np.sum(algoritmo.clusters != -1))/algoritmo.getNumClusters()
    else:
        media_puntos_cluster = 0

    instancias_por_cluster = [0] * algoritmo.getNumClusters()
    for cluster in algoritmo.clusters:
        if cluster != -1:
            instancias_por_cluster[cluster] += 1
    minimo_instancias = 0
    if len(instancias_por_cluster)!=0:
        minimo_instancias = min(instancias_por_cluster)

        # Devuelve el número de instancias de ruido (puedes usar otra métrica)

    if optunaNCluster <= 1:
        return -1, 10000, 0, -sys.maxsize
    else:
        s = silhouette_score(loadedEmbedding, algoritmo.clusters)
        return s, optunaNCluster, media_puntos_cluster, minimo_instancias


def barridoDBSCANOPtuna(nInstances, dimension, type):
    loadedEmbedding = loadEmbeddings(length=nInstances, dimension=dimension, type=type)
    # Optimiza para minimizar el ruido
    study = optuna.create_study(directions=['maximize', 'minimize', 'maximize', 'maximize'])

    # Realiza la optimización de los parámetros
    study.optimize(lambda trial: objective(trial, loadedEmbedding), n_trials=100)

    # Obtiene los mejores parámetros encontrados
    best_trial = max(study.best_trials, key=lambda t: t.values[1])
    best_epsilon = best_trial.params['epsilon']
    best_minPt = best_trial.params['minPt']

    best_silhouette, optunaNCluster, best_media_puntos_cluster, best_minimo_instancias = best_trial.values


    saveInCSV2(nInstances=nInstances,
              dimension=dimension,
              espilon=best_epsilon,
              minPts=best_minPt,
              media_puntos_cluster=best_media_puntos_cluster,
              minimo_instancias=best_minimo_instancias,
              nClusters=optunaNCluster,
              silhouette=best_silhouette)


if __name__ == '__main__':
    nInstaces = int(sys.argv[2])
    dimension = int(sys.argv[3])
    type = sys.argv[4]

    if sys.argv[1] == 'optuna':
        barridoDBSCANOPtuna(nInstances=nInstaces, dimension=dimension, type=type)
    elif sys.argv[1] == 'heuristic':
        heuristicoEpsilonDBSCAN(nInstances=nInstaces, dimension=dimension, type=type)
    elif sys.argv[1] == 'barridoDBSCAN':
        barridoDBSCAN(nInstances=nInstaces,
                      dimension=dimension,
                      type=type,
                      espilonList=[0.05, 1, 2, 3, 4, 5, 10, 20, 50, 100, 500],
                      minPtsList=[25, 50, 75, 100, 125, 150, 175, 200])
    else:
        print('Error in passing arguments')
        print('Execution example: paramOptimization.py optuna 10000 768')
