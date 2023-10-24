from loadSaveData import loadRAW
from tokenization import tokenize
import vectorization
import clustering
import evaluation

# Parameters
path = '../Datasets/Suicide_Detection10000.csv'
dimensions = 500
vectorizationMode = vectorization.doc2vec # doc2vec, tfidf, bertTransformer
clusteringAlgorithm = clustering.DensityAlgorithmUrruela # DensityAlgorithmUrruela, DensityAlgorithm, DensityAlgorithm2, DBScanOriginal
epsilon = 4
minPts = 5

# PreProcessing
rawData = loadRAW(path)
if vectorizationMode != vectorization.bertTransformer:
    textosToken = tokenize(rawData)
    textosEmbedding = vectorizationMode(textosToken=textosToken, dimensiones=dimensions)
else: 
    textEmbeddings = vectorizationMode(rawData)

# Clustering
algoritmo = clusteringAlgorithm(textEmbeddings, epsilon=epsilon, minPt=minPts, dim=dimensions)
algoritmo.ejecutarAlgoritmo()
algoritmo.imprimir()
clusters = algoritmo.clusters

# Evaluation
evaluation.classToCluster(rawData, clusters)
evaluation.wordCloud(clusters, textosToken)
