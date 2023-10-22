import pandas as pd
import spacy
import emoji
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm
import time

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import get_tmpfile

from loadSaveData import saveEmbeddings, saveTokens, loadEmbeddings, loadTokens
from results import classToCluster, wordCloud


def distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)


class PreProcessing:

    def __init__(self, path, dim):
        self.datasetPath = path
        self.textos = []
        self.textos_token = []
        self.documentVectors = []
        self.data = None
        self.dimensiones = dim

    def cargarDatos(self):

        self.data = pd.read_csv(self.datasetPath)

        for instancia in self.data.values:
            self.textos.append(instancia[1])

    def limpiezaDatos(self):

        nlp = spacy.load("en_core_web_sm")  # Cargar modelo
        nlp.add_pipe("emoji", first=True)

        for texto in tqdm(self.textos, desc="Procesando textos"):
            texto = emoji.demojize(texto)  # Emojis a texto
            texto = texto.replace(':', ' ').replace('filler', ' ').replace('filer', ' ').replace('_', ' ')
            doc = nlp(texto)
            lexical_tokens = [token.lemma_.lower() for token in doc if len(token.text) > 3 and token.is_alpha and not token.is_stop]
            self.textos_token.append(lexical_tokens)

        saveTokens(self.textos_token)

    def doc2vec(self):

        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(self.textos_token)]
        model = Doc2Vec(documents, vector_size=self.dimensiones, window=2, dm=1, epochs=100, workers=4)

        model.build_vocab(documents)
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

        model.save(get_tmpfile("my_doc2vec_model"))

        self.documentVectors = [model.infer_vector(doc) for doc in self.textos_token]

        saveEmbeddings(self.documentVectors, self.dimensiones)


class DensityAlgorithm:

    def __init__(self, vectors, epsilon, minPt):
        self.vectors = vectors  # DOCUMENTOS VECTORIZADOS
        self.epsilon = epsilon  # RADIO PARA CONSIDERAR VECINOS
        self.minPt = minPt  # MINIMO DE VECINOS PARA CONSIDERAR NUCLEO

        self.vecinos = []
        self.nucleos = []  # CONJUNTO DE INSTANCIAS NUCLEO
        self.clusters = [-1] * len(self.vectors) # CONJUNTO DE CLUSTERS
        self.distancias = {} # DISTANCIAS ENTRE VECTORES {frozenSet: float}

    def ejectuarAlgoritmo(self):
        self.calcular_distancias()
        self.buscarNucleos()
        self.crearClusters()

    def buscarNucleos(self):
        for i, _ in enumerate(self.vectors):
            v = []
            for j, _ in enumerate(self.vectors):
                if j != i and self.distancias.get('_'.join(sorted([str(i), str(j)]))) <= self.epsilon:
                    v.append(j)
            self.vecinos.append(v)
            if len(v) >= self.minPt:
                self.nucleos.append(i)

    def crearClusters(self):
        numCluster = -1
        nucleosPorVisitar = []
        for i in self.nucleos:
            if self.clusters[i] == -1:
                numCluster += 1
                self.clusters[i] = numCluster
                nucleosPorVisitar.append(i)

                while nucleosPorVisitar:
                    j = nucleosPorVisitar.pop()

                    for index in self.vecinos[j]:
                        if self.clusters[index] == -1:
                            self.clusters[index] = numCluster
                            if index in self.nucleos:
                                nucleosPorVisitar.append(index)

    def calcular_distancias(self):
        for i, doc in enumerate(self.vectors):
            for j, doc2 in enumerate(self.vectors):
                if j != i:
                    if (pair := '_'.join(sorted([str(i), str(j)]))) not in self.distancias:
                        distEuc = np.linalg.norm(doc - doc2)
                        self.distancias[pair] = distEuc
        print(f'LAS DISTANCIAS SUMAN: {len(self.distancias)}')

    def imprimir(self):
        total = 0
        for cluster in range(min(self.clusters), max(self.clusters) + 2):
            kont = 0
            for i in self.clusters:
                if i == cluster:
                    kont += 1
            if cluster == -1:
                print(f'Hay un total de {kont} instancias que son ruido')
            else:
                print(f'Del cluster {cluster} hay {kont} instancias')
            total = total + kont


class DensityAlgorithm2:
    def __init__(self, vectors, epsilon, minPt):
        self.vectors = vectors  # DOCUMENTOS VECTORIZADOS
        self.epsilon = epsilon  # RADIO PARA CONSIDERAR VECINOS
        self.minPt = minPt  # MINIMO DE VECINOS PARA CONSIDERAR NUCLEO
        self.clusters = []

    def get_neighbors(self, point):
        neighbors = []
        # i = indice de cada dato
        # d = vetor de cada dato
        for i, d in enumerate(self.vectors):
            if distance(point, d) <= self.epsilon:
                neighbors.append(i)
        return neighbors

    def expand_cluster(self, labels, i, neighbors, cluster_id):
        labels[i] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor = neighbors[i]
            if labels[neighbor] == -1:
                labels[neighbor] = cluster_id
            elif labels[neighbor] == 0:
                labels[neighbor] = cluster_id
                new_neighbors = self.get_neighbors(self.vectors[neighbor])
                if len(new_neighbors) >= self.minPt:
                    neighbors = neighbors + new_neighbors
            i += 1

    def ejecutarAlgoritmo(self):
        labels = [0] * len(self.vectors)
        cluster_id = 0
        for i in range(len(self.vectors)):
            if labels[i] == 0:
                neighbors = self.get_neighbors(self.vectors[i])
                if len(neighbors) < self.minPt:
                    labels[i] = -1
                else:
                    cluster_id = cluster_id + 1
                    self.expand_cluster(labels, i, neighbors, cluster_id)
        self.clusters = labels
        return labels

    def imprimir(self):
        total = 0
        for cluster in range(min(self.clusters), max(self.clusters) + 1):
            kont = 0
            for i in self.clusters:
                if i == cluster:
                    kont += 1
            if cluster == -1:
                print(f'Hay un total de {kont} instancias que son ruido')
            else:
                print(f'Del cluster {cluster} hay {kont} instancias')
            total = total + kont

class DBScanOriginal:
    def __init__(self, vectors, epsilon, minPt):
        self.vectors = vectors  # DOCUMENTOS VECTORIZADOS
        self.epsilon = epsilon  # RADIO PARA CONSIDERAR VECINOS
        self.minPt = minPt  # MINIMO DE VECINOS PARA CONSIDERAR NUCLEO
        self.clusters = []
        self.numClusters = None

    def ejecutarAlgoritmo(self):
        # Aplicar DBSCAN a los vectores de documentos
        dbscan = DBSCAN(eps=self.epsilon, min_samples=self.minPt)  # Ajusta los parámetros según tu caso
        self.clusters = dbscan.fit_predict(self.vectors)

    def imprimir(self):
        total = 0
        for cluster in range(min(self.clusters), max(self.clusters) + 1):
            kont = 0
            for i in self.clusters:
                if i == cluster:
                    kont += 1
            if cluster == -1:
                print(f'Hay un total de {kont} instancias que son ruido')
            else:
                print(f'Del cluster {cluster} hay {kont} instancias')
            total = total + kont

    def getNoiseInstances(self):
        return np.count_nonzero(self.clusters == -1)

    def getNumClusters(self):
        if self.numClusters is not None: return self.numClusters
        else:
            self.numClusters = len(set(self.clusters) - {-1})
            return self.numClusters

def llamar_al_metodo(metodo, preProcess, epsilon, minPt):
    if metodo == 0:
        # CLUSTERING DBSCAN ORIGINAL
        start_time = time.time()
        algoritmo3 = DBScanOriginal(preProcess.documentVectors, epsilon=epsilon, minPt=minPt)
        algoritmo3.ejecutarAlgoritmo()
        end_time = time.time()

        print(f'\n\n\nTiempo DBScan: {end_time - start_time}')

        algoritmo3.imprimir()
        classToCluster(preProcess.data, algoritmo3.clusters)
        wordCloud(algoritmo3.clusters, preProcess.textos_token)

    elif metodo == 1:
        # CLUSTERING DBSCAN IMPLEMENTADO
        start_time = time.time()
        algoritmo2 = DensityAlgorithm2(preProcess.documentVectors, epsilon=epsilon, minPt=minPt)
        algoritmo2.ejecutarAlgoritmo()
        end_time = time.time()

        print(f'\n\n\nTiempo Nagore: {end_time - start_time}')
        algoritmo2.imprimir()

        classToCluster(preProcess.data, algoritmo2.clusters)
        wordCloud(algoritmo2.clusters, preProcess.textos_token)

    elif metodo == 2:
        # CLUSTERING ALTERNATIVO
        start_time = time.time()
        algoritmo = DensityAlgorithm(preProcess.documentVectors, epsilon=epsilon, minPt=minPt)
        algoritmo.ejectuarAlgoritmo()
        end_time = time.time()

        print(f'\n\n\nTiempo Maitane: {end_time - start_time}')
        algoritmo.imprimir()

        classToCluster(preProcess.data, algoritmo.clusters)
        wordCloud(algoritmo.clusters, preProcess.textos_token)

    elif metodo == 3:
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        # Etiquetas predichas para cada instancia de X_train:
        kmeansLabels = kmeans.fit_predict(preProcess.documentVectors)
        classToCluster(preProcess.data, kmeansLabels)
        wordCloud(kmeansLabels, preProcess.textos_token)
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

        algoritmo3 = DBScanOriginal(x_train, epsilon=epsilon, minPt=minPt)
        algoritmo3.ejecutarAlgoritmo()

        algoritmo3.imprimir()
        # classToCluster(preProcess.data, algoritmo3.clusters)
        # wordCloud(algoritmo3.clusters, preProcess.textos_token)
    else:
        pass



if __name__ == '__main__':
    # PREPROCESADO DE DATOS


    preProcess = PreProcessing('../Datasets/Suicide_Detection10000.csv', 150)
    preProcess.cargarDatos()
    preProcess.limpiezaDatos()
    preProcess.doc2vec()

    #documentVectors = TSNE(n_components=2, random_state=0).fit_transform(documentVectors)

    # PROCESO DE CLUSTERING
    # PARAMETROS:
    epsilon = 7
    minPt = 2
    # preProcess.documentVectors = loadEmbeddings(10000, 150)
    # preProcess.textos_token = loadTokens(10000)
    #llamar_al_metodo(0, preProcess, epsilon, minPt) # DBSCAN
    # llamar_al_metodo(1, preProcess, epsilon, minPt) # NAGORE
    #llamar_al_metodo(2, preProcess, epsilon, minPt) # MAITANE
    #llamar_al_metodo(3, preProcess, epsilon, minPt) # KMEANS
    llamar_al_metodo(4, None, epsilon, minPt) # MNIST








