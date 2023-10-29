from loadSaveData import loadRAW, saveClusters, loadEmbeddings, loadSinLimpiarTokens
from tokenization import tokenize
import vectorization
import clustering
import evaluation
import sys


def preProcess(nInstances, vectorsDimension, vectorizationMode):
    path = f'../Datasets/Suicide_Detection{nInstances}.csv' # Previosuly reduced with reduceDataset.py
    rawData = loadRAW(path)
    if vectorizationMode != vectorization.bertTransformer:
        textosToken = tokenize(rawData)
        textEmbeddings = vectorizationMode(textosToken=textosToken, dimensiones=vectorsDimension)
    else: 
        textEmbeddings = vectorizationMode(rawData)
    return rawData, textEmbeddings


def executeClustering(clusteringAlgorithm, epsilon, minPts):
    algoritmo = clusteringAlgorithm(textEmbeddings, epsilon=epsilon, minPt=minPts)
    algoritmo.ejecutarAlgoritmo()
    algoritmo.imprimir()
    clusters = algoritmo.clusters
    numClusters = algoritmo.getNumClusters()
    return clusters, numClusters


def evaluate(nInstances, rawData, clusters, numClusters):
    tokensSinLimpiar = loadSinLimpiarTokens(length=nInstances)

    evaluation.classToCluster(rawData, clusters)
    evaluation.wordCloud(clusters, tokensSinLimpiar)
    evaluation.getClusterSample(clusterList=clusters, 
                                numClusters=numClusters,
                                rawData=rawData,
                                sample=5)


if __name__ == '__main__':
    nInstances = int(sys.argv[1])
    vectorsDimension = int(sys.argv[2])
    if sys.argv[3] == 'doc2vec':
        vectorizationMode = vectorization.bertTransformer
    elif sys.argv[3] == 'tfidf':
        vectorizationMode = vectorization.tfidf
    elif sys.argv[3] == 'bert':
        vectorizationMode = vectorization.bertTransformer
    if sys.argv[4] == 'ourDensityAlgorithm':
        clusteringAlgorithm = clustering.DensityAlgorithmUrruela
    elif sys.argv[4] == 'dbscan':
        clusteringAlgorithm = clustering.DBScanOriginal
    epsilon = int(sys.argv[5])
    minPts = int(sys.argv[6])

    rawData, textEmbeddings = preProcess(nInstances, vectorsDimension, vectorizationMode)
    clusters, numClusters = executeClustering(clusteringAlgorithm, epsilon, minPts)
    evaluate(nInstances, rawData, clusters, numClusters)
