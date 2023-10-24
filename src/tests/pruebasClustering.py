import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from tqdm import tqdm
import time

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import get_tmpfile
from sklearn.metrics.pairwise import cosine_similarity

from loadSaveData import saveEmbeddings, saveTokens, loadEmbeddings, loadTokens, loadRAW
from evaluation import classToCluster, wordCloud
from clustering import DBScanOriginal, DensityAlgorithm, DensityAlgorithm2
from tokenization import tokenize
from vectorization import doc2vec

def llamar_al_metodo(metodo, epsilon, minPt, rawData, textTokens, textEmbeddings):
    if metodo == 0:
        # CLUSTERING DBSCAN ORIGINAL
        start_time = time.time()
        algoritmo3 = DBScanOriginal(textEmbeddings, epsilon=epsilon, minPt=minPt)
        algoritmo3.ejecutarAlgoritmo()
        end_time = time.time()

        print(f'\n\n\nTiempo DBScan: {end_time - start_time}')

        algoritmo3.imprimir()
        classToCluster(rawData, algoritmo3.clusters)
        wordCloud(algoritmo3.clusters, textTokens)

    elif metodo == 1:
        # CLUSTERING DBSCAN IMPLEMENTADO
        start_time = time.time()
        algoritmo2 = DensityAlgorithm2(textEmbeddings, epsilon=epsilon, minPt=minPt)
        algoritmo2.ejecutarAlgoritmo()
        end_time = time.time()

        print(f'\n\n\nTiempo Nagore: {end_time - start_time}')
        algoritmo2.imprimir()

        classToCluster(rawData, algoritmo2.clusters)
        wordCloud(algoritmo2.clusters, textTokens)

    elif metodo == 2:
        # CLUSTERING ALTERNATIVO
        start_time = time.time()
        algoritmo = DensityAlgorithm(textEmbeddings, epsilon=epsilon, minPt=minPt)
        algoritmo.ejectuarAlgoritmo()
        end_time = time.time()

        print(f'\n\n\nTiempo Maitane: {end_time - start_time}')
        algoritmo.imprimir()

        classToCluster(rawData, algoritmo.clusters)
        wordCloud(algoritmo.clusters, textTokens)

    elif metodo == 3:
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        # Etiquetas predichas para cada instancia de X_train:
        kmeansLabels = kmeans.fit_predict(textEmbeddings)
        classToCluster(rawData, kmeansLabels)
        wordCloud(kmeansLabels, textTokens)
        total = 0
        for cluster in range(min(kmeansLabels), max(kmeansLabels) + 1):
            kont = 0
            for i in kmeansLabels:
                if i == cluster:
                    kont += 1
            if cluster == -1:
                print(f'Hay un total de {kont} instancias que son ruido')
            else:
                print(f'Del cluster {cluster} hay {kont} instancias')
            total = total + kont
    elif metodo == 4:
        x_train = pd.read_csv('../Datasets/mnist_train.csv')
        y_train = x_train['label'].copy()
        x_train.drop('label', axis=1, inplace=True)

        algoritmo3 = DBScanOriginal(textEmbeddings, epsilon=epsilon, minPt=minPt, metric='cosine')
        algoritmo3.ejecutarAlgoritmo()

        algoritmo3.imprimir()
        # classToCluster(preProcess.data, algoritmo3.clusters)
        # wordCloud(algoritmo3.clusters, preProcess.textos_token)
    else:
        pass



if __name__ == '__main__':
    # PREPROCESADO DE DATOS

    rawData = loadRAW('../../Datasets/Suicide_Detection_10000.csv')
    textTokens = tokenize(rawData)
    textEmbeddings = doc2vec(textTokens, 150)

    #documentVectors = TSNE(n_components=2, random_state=0).fit_transform(documentVectors)

    # PROCESO DE CLUSTERING
    # PARAMETROS:
    epsilon =22.27
    minPt = 1
    # preProcess.documentVectors = loadEmbeddings(10000, 150)
    # preProcess.textos_token = loadTokens(10000)
    llamar_al_metodo(0, epsilon, minPt, rawData, textTokens, textEmbeddings) # DBSCAN
    # llamar_al_metodo(1, preProcess, epsilon, minPt) # NAGORE
    llamar_al_metodo(2, epsilon, minPt, rawData, textTokens, textEmbeddings) # MAITANE
    #llamar_al_metodo(3, preProcess, epsilon, minPt) # KMEANS
    #llamar_al_metodo(4, None, epsilon, minPt) # MNIST