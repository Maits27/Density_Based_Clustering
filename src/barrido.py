import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from main import DensityAlgorithm, DBScanOriginal
import numpy as np
from loadSaveData import loadEmbeddings
import csv
import optuna
import plotly.express as px

optunaNCluster = 0


def heuristicoEpsilonDBSCAN(nInstances, dimension):
    # Load vectors
    embeddingVectors = loadEmbeddings(length=nInstances, dimension=dimension)

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


def distance_distribution(nInstances, dimension):
    pares_calculados = set()
    distancias = []
    ema =  [0] * 10
    embeddingVectors = loadEmbeddings(length=nInstances, dimension=dimension)
    for i, doc in enumerate(embeddingVectors):
        for j, doc2 in enumerate(embeddingVectors):
            if j != i:
                if (pair := '_'.join(sorted([str(i), str(j)]))) not in pares_calculados:
                    distancias.append(np.linalg.norm(doc - doc2))
                    pares_calculados.add(pair)
    for dist in distancias:
        index = int(dist/5)
        if index>10: ema[9] = ema[9]+1
        else: ema[index] = ema[index]+1
    print(f'LOS TARTES DE LAS DISTANCIAS: {ema}')
    fig = px.histogram(x=distancias, nbins=20)
    fig.show()

def barridoDBSCAN(nInstances, dimension, espilonList, minPtsList):
    # Load vectors
    embeddingVectors = loadEmbeddings(length=nInstances, dimension=dimension)

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


def saveInCSV(nInstances, dimension, espilon, minPts, nClusters, silhouette):
    with open('../Barridos_7.csv', 'a') as file:
        writer = csv.writer(file, delimiter='|')
        writer.writerow([nInstances, dimension, espilon, minPts, nClusters, silhouette])

def saveInCSV2(nInstances, dimension, espilon, minPts, media_puntos_cluster, nClusters, silhouette):
    with open('../Barridos_7.csv', 'w', encoding='utf8') as file:
        file.write('N_Instances\tDim\tEps\tminPts\tmediaPuntosCluster\tnClusters\tMetric\n')
        file.write(f'{nInstances}\t{dimension}\t{espilon}\t{minPts}\t{media_puntos_cluster}\t{nClusters}\t{silhouette}')


def objective(trial, loadedEmbedding):
    epsilon = trial.suggest_float('epsilon', 5, 20.0, step=0.001)
    minPt = trial.suggest_int('minPt', 1, 30)


    # Utiliza los valores sugeridos por Optuna para la ejecución
    algoritmo = DBScanOriginal(loadedEmbedding, epsilon=epsilon, minPt=minPt)
    algoritmo.ejecutarAlgoritmo()

    optunaNCluster = algoritmo.getNumClusters()

    # TODO: Calcular media puntos cluster

    if algoritmo.getNumClusters() > 1:
        media_puntos_cluster = (np.sum(algoritmo.clusters != -1))/algoritmo.getNumClusters()
    else:
        media_puntos_cluster = 0

        # Devuelve el número de instancias de ruido (puedes usar otra métrica)
    if optunaNCluster <= 1:
        return -1, 10000, 0
    else:
        return silhouette_score(loadedEmbedding, algoritmo.clusters), optunaNCluster, media_puntos_cluster


def barridoDBSCANOPtuna(nInstances, dimension):
    loadedEmbedding = loadEmbeddings(length=nInstances, dimension=dimension)

    # Optimiza para minimizar el ruido
    study = optuna.create_study(directions=['maximize', 'minimize', 'maximize'])

    # Realiza la optimización de los parámetros
    study.optimize(lambda trial: objective(trial, loadedEmbedding), n_trials=1000)

    # Obtiene los mejores parámetros encontrados
    best_trial = max(study.best_trials, key=lambda t: t.values[1])
    best_epsilon = best_trial.params['epsilon']
    best_minPt = best_trial.params['minPt']

    best_silhouette, optunaNCluster, best_media_puntos_cluster,  = best_trial.values


    saveInCSV2(nInstances=nInstances,
              dimension=dimension,
              espilon=best_epsilon,
              minPts=best_minPt,
              media_puntos_cluster=best_media_puntos_cluster,
              nClusters=optunaNCluster,
              silhouette=best_silhouette)


barridoDBSCANOPtuna(nInstances=10000, dimension=150)

# print(len(loadEmbeddings(length=1000, dimension=150)))
# barridoDBSCAN(nInstances=1000,
#               dimension=150,
#               espilonList=[0.05, 1, 2, 3, 4, 5, 10, 20, 50, 100, 500],
#               minPtsList=[25, 50, 75, 100, 125, 150, 175, 200])
# heuristicoEpsilonDBSCAN(nInstances=20000, dimension=150)
#distance_distribution(nInstances=10000, dimension=150)
