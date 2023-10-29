from loadSaveData import loadRAW, loadSinLimpiarTokens, loadRAWwithClass
from tokenization import tokenize, tokenizarSinLimpiar
import vectorization
import clustering
import evaluation
import sys


def preProcess(nInstances, vectorsDimension, vectorizationMode):
    path = f'../Datasets/Suicide_Detection{nInstances}.csv' # Previosuly reduced with reduceDataset.py
    rawData = loadRAW(path)
    rawDataWithClass = loadRAWwithClass(path)
    if vectorizationMode != vectorization.bertTransformer:
        textosToken = tokenize(rawData)
        textEmbeddings = vectorizationMode(textosToken=textosToken, dimensiones=vectorsDimension)
    else: 
        textEmbeddings = vectorizationMode(rawData)
    return rawData, rawDataWithClass, textEmbeddings


def executeClustering(clusteringAlgorithm, epsilon, minPts):
    algoritmo = clusteringAlgorithm(textEmbeddings, epsilon=epsilon, minPt=minPts)
    algoritmo.ejecutarAlgoritmo()
    algoritmo.imprimir()
    clusters = algoritmo.clusters
    numClusters = algoritmo.getNumClusters()
    return clusters, numClusters


def evaluate(rawData, rawDataWithClass, clusters, numClusters):
    tokensSinLimpiar = tokenizarSinLimpiar(rawDataWithClass) 

    evaluation.classToCluster(tokensSinLimpiar, clusters) # TODO da error, no hay clase
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
    epsilon = float(sys.argv[5])
    minPts = int(sys.argv[6])

    rawData, rawDataWithClass, textEmbeddings = preProcess(nInstances, vectorsDimension, vectorizationMode)
    clusters, numClusters = executeClustering(clusteringAlgorithm, epsilon, minPts)
    evaluate(rawData, rawDataWithClass, clusters, numClusters)
