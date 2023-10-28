from loadSaveData import loadRAW, saveClusters, loadEmbeddings, loadSinLimpiarTokens
from tokenization import tokenize
import vectorization
import clustering
import evaluation

# Definir parámetros
nInstances = 10000
path = f'../Datasets/Suicide_Detection{nInstances}.csv' # Previosuly reduced with reduceDataset.py
vectorsDimension = 500 # Not used for 'bertTransformer'
vectorizationMode = vectorization.bertTransformer # doc2vec, tfidf, bertTransformer
clusteringAlgorithm = clustering.DensityAlgorithmUrruela # DensityAlgorithmUrruela, DensityAlgorithm, DensityAlgorithm2, DBScanOriginal
epsilon = 2.567
minPts = 12

# Pre-proceso
rawData = loadRAW(path)
if vectorizationMode != vectorization.bertTransformer:
    textosToken = tokenize(rawData)
    textosEmbedding = vectorizationMode(textosToken=textosToken, dimensiones=vectorsDimension)
else: 
    textEmbeddings = vectorizationMode(rawData)

# Clustering
algoritmo = clusteringAlgorithm(textEmbeddings, epsilon=epsilon, minPt=minPts)
algoritmo.ejecutarAlgoritmo()
algoritmo.imprimir()
clusters = algoritmo.clusters
saveClusters(clusters, 'dbscan')

# Evaluación
tokensSinLimpiar = loadSinLimpiarTokens(length=nInstances)

evaluation.classToCluster(rawData, clusters)
evaluation.wordCloud(clusters, tokensSinLimpiar)
evaluation.getClusterSample(clusterList=algoritmo.clusters, 
                            numClusters=algoritmo.getNumClusters(),
                            rawData=rawData,
                            sample=5)
