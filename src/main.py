from loadSaveData import loadRAW, saveClusters,loadClusters, loadEmbeddings
from tokenization import tokenize
import vectorization
import clustering
import evaluation

# Parameters
path = '../Datasets/Suicide_Detection10000.csv'
dimensions = 500
vectorizationMode = vectorization.doc2vec # doc2vec, tfidf, bertTransformer
clusteringAlgorithm = clustering.DBScanOriginal # DensityAlgorithmUrruela, DensityAlgorithm, DensityAlgorithm2, DBScanOriginal
epsilon = 0.05
minPts = 3

# PreProcessing
rawData = loadRAW(path)
if vectorizationMode != vectorization.bertTransformer:
    textosToken = tokenize(rawData)
    textosEmbedding = vectorizationMode(textosToken=textosToken, dimensiones=dimensions)
else: 
    textEmbeddings = vectorizationMode(rawData)

# Clustering
algoritmo = clusteringAlgorithm(textEmbeddings, epsilon=epsilon, minPt=minPts)
algoritmo.ejecutarAlgoritmo()
algoritmo.imprimir()
clusters = algoritmo.clusters
saveClusters(clusters,'dbscan')

# Evaluation
evaluation.classToCluster(rawData, clusters)
evaluation.wordCloud(clusters, textosToken)
