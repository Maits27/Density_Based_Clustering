import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from main import DensityAlgorithm, DBScanOriginal
import numpy as np
from loadSaveData import loadEmbeddings
import csv
import optuna

optunaNCluster = 0

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
    with open('../Barridos.csv', 'a') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow([nInstances, dimension, espilon, minPts, nClusters, silhouette])


def objective(trial, loadedEmbedding):    
    epsilon = trial.suggest_float('epsilon', 0.1, 10.0)
    minPt = trial.suggest_int('minPt', 150, 300)
    
    # Utiliza los valores sugeridos por Optuna para la ejecución
    algoritmo = DBScanOriginal(loadedEmbedding, epsilon=epsilon, minPt=minPt)
    algoritmo.ejecutarAlgoritmo()

    optunaNCluster = algoritmo.getNumClusters()
    
    # Devuelve el número de instancias de ruido (puedes usar otra métrica)
    if algoritmo.getNumClusters() <= 1: return 0
    else: return silhouette_score(loadedEmbedding, algoritmo.clusters)
    
    
def barridoDBSCANOPtuna(nInstances, dimension):
    loadedEmbedding = loadEmbeddings(length=nInstances, dimension=dimension)

    # Optimiza para minimizar el ruido
    study = optuna.create_study(direction='maximize')  
    
    # Realiza la optimización de los parámetros
    study.optimize(lambda trial: objective(trial, loadedEmbedding), n_trials=1000)
    
    # Obtiene los mejores parámetros encontrados
    best_epsilon = study.best_params['epsilon']
    best_minPt = study.best_params['minPt']
    best_silhouette = study.best_value

    saveInCSV(nInstances=nInstances, 
              dimension=dimension, 
              espilon=best_epsilon, 
              minPts=best_minPt, 
              nClusters=optunaNCluster,
              silhouette=best_silhouette)


#barridoDBSCANOPtuna(nInstances=1000, dimension=150)
print(len(loadEmbeddings(length=1000, dimension=150)))
barridoDBSCAN(nInstances=1000, 
              dimension=150, 
              espilonList=[0.05, 1, 2, 3, 4, 5, 10, 20, 50, 100, 500], 
              minPtsList=[25, 50, 75, 100, 125, 150, 175, 200])
#heuristicoEpsilonDBSCAN(nInstances=20000, dimension=150)