import pandas as pd
import spacy
import emoji
import numpy as np
from tqdm import tqdm

from sklearn.cluster import DBSCAN
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import get_tmpfile


def distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)


class PreProcessing:

    def __init__(self, path):
        self.datasetPath = path
        self.textos = []
        self.textos_token = []
        self.documentVectors = []

    def cargarDatos(self):

        data = pd.read_csv(self.datasetPath)

        for instancia in data.values:
            self.textos.append(instancia[1])

    def limpiezaDatos(self):

        nlp = spacy.load("en_core_web_sm")  # Cargar modelo
        nlp.add_pipe("emoji", first=True)

        for texto in tqdm(self.textos, desc="Procesando textos"):
            texto = emoji.demojize(texto)  # Emojis a texto
            texto = texto.replace(':', ' ').replace('filler', ' ').replace('filer', ' ').replace('_', ' ')
            doc = nlp(texto)
            lexical_tokens = [token.lemma_.lower() for token in doc if len(token.text) > 3 and token.is_alpha]
            self.textos_token.append(lexical_tokens)

    def doc2vec(self):

        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(self.textos_token)]
        model = Doc2Vec(documents, vector_size=150, window=2, dm=1, epochs=100, workers=4)

        model.build_vocab(documents)
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

        model.save(get_tmpfile("my_doc2vec_model"))

        self.documentVectors = [model.infer_vector(doc) for doc in self.textos_token]


class DensityAlgorithm:

    def __init__(self, vectors, epsilon, minPt):
        self.vectors = vectors  # DOCUMENTOS VECTORIZADOS
        self.epsilon = epsilon  # RADIO PARA CONSIDERAR VECINOS
        self.minPt = minPt  # MINIMO DE VECINOS PARA CONSIDERAR NUCLEO

        self.vecinos = []
        self.nucleos = []  # CONJUNTO DE INSTANCIAS NUCLEO
        self.clusters = []  # CONJUNTO DE CLUSTERS
        self.clustersValidos = []  # CONJUNTO DE CLUSTERS SELECCIONADOS
        self.alcanzables = []
        self.numCluster = -1

    def ejectuarAlgoritmo(self):
        self.buscarNucleos()
        self.crearClusters()
        self.seleccionClusters()
        self.reclasificarInst()
        self.reasignarLabelCluster()

    def buscarNucleos(self):
        for i, doc in enumerate(self.vectors):
            v = []
            for j, doc2 in enumerate(self.vectors):
                distEuc = np.linalg.norm(doc - doc2)
                if distEuc <= self.epsilon:
                    v.append((j, doc2))
            self.vecinos.append(v)
            if len(v) >= self.minPt:
                self.nucleos.append((i, doc))

    def crearClusters(self):
        for i in tqdm(range(len(self.vectors)), desc="Creando Clusters"):
            self.clusters.append(-1)

        nucleosPorVisitar = []
        for i, nucleo in self.nucleos:
            if self.clusters[i] == -1:
                self.numCluster += 1
                self.clusters[i] = self.numCluster
                nucleosPorVisitar.append((i, nucleo))
                while nucleosPorVisitar:
                    j, nucleo_actual = nucleosPorVisitar.pop()
                    for index, vecino in self.vecinos[j]:
                        if self.clusters[index] == -1:
                            self.clusters[index] = self.numCluster
                            if (j, nucleo_actual) in self.nucleos:
                                nucleosPorVisitar.append((j, nucleo_actual))

    def seleccionClusters(self):
        for c in range(self.numCluster + 1):
            if self.clusters.count(c) < self.minPt:
                for i, clus in enumerate(self.clusters):
                    if clus == c:
                        self.alcanzables.append((i, self.vecinos[i]))
            else:
                self.clustersValidos.append(c)

    def reclasificarInst(self):
        for i, vecinos in self.alcanzables:
            previo = self.clusters[i]
            for j, v in vecinos:
                if self.clusters[j] in self.clustersValidos:
                    self.clusters[i] = self.clusters[j]
            if previo == self.clusters[i]:
                self.clusters[i] = -1

    def reasignarLabelCluster(self):
        for i in range(len(self.clustersValidos)):
            for j in range(len(self.clusters)):
                if self.clusters[j] == self.clustersValidos[i]:
                    self.clusters[j] = i

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
        labels[i]=cluster_id
        i=0
        while i< len(neighbors):
            neighbor=neighbors[i]
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
        self.clusters=labels
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


if __name__ == '__main__':
    # PREPROCESADO DE DATOS

    preProcess = PreProcessing('../Datasets/corto.csv')
    preProcess.cargarDatos()
    preProcess.limpiezaDatos()
    preProcess.doc2vec()

    # PROCESO DE CLUSTERING

    algoritmo = DensityAlgorithm(preProcess.documentVectors, epsilon=5, minPt=10)
    algoritmo.ejectuarAlgoritmo()

    print(algoritmo.clusters)
    algoritmo.imprimir()

    algoritmo2 = DensityAlgorithm2(preProcess.documentVectors, epsilon=5, minPt=10)
    algoritmo2.ejecutarAlgoritmo()

    print(algoritmo2.clusters)
    algoritmo2.imprimir()
