import time
from clustering import DensityAlgorithmUrruela
from evaluation import classToCluster, wordCloud
from loadSaveData import loadRAW, loadEmbeddings
from tokenization import tokenize
from vectorization import doc2vec

def llamar_al_metodo(rawData, textTokens, textEmbeddings, dimensions,  epsilon, minPt):

    # CLUSTERING ALTERNATIVO
    start_time = time.time()
    algoritmo = DensityAlgorithmUrruela(textEmbeddings, epsilon=epsilon, minPt=minPt, dim=dimensions)
    algoritmo.ejecutarAlgoritmo()
    end_time = time.time()

    print(f'\n\n\nTiempo Maitane: {end_time - start_time}')
    algoritmo.imprimir()

    classToCluster(rawData, algoritmo.clusters)
    wordCloud(algoritmo.clusters, textTokens)




if __name__ == '__main__':
    start_time = time.time()
    # PREPROCESADO DE DATOS
    rawData = loadRAW('../Datasets/Suicide_Detection10000.csv')

    # PREPROCESO SIN HACER
    textTokens = tokenize(rawData)
    #textEmbeddings = doc2vec(textTokens, 150)

    # PREPROCESADO HECHO:
    textEmbeddings = loadEmbeddings(10000, 150)

    #preProcess.textos_token = loadTokens(10000)
    end_time = time.time()
    print(f'\n\n\nTiempo preproceso: {end_time - start_time}')

    # PROCESO DE CLUSTERING
    # PARAMETROS:
    epsilon = 12.25#15#10
    minPt = 10#10#5
    llamar_al_metodo(rawData, textTokens, textEmbeddings, 150, epsilon, minPt) # MAITANE