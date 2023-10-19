import pandas as pd
import spacy
import emoji
import numpy as np
from tqdm import tqdm

from sklearn.cluster import DBSCAN
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import get_tmpfile


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



if __name__ == '__main__':

    # PREPROCESADO DE DATOS

    preProcess = PreProcessing('../Datasets/Suicide_Detection20000.csv')
    preProcess.cargarDatos()
    preProcess.limpiezaDatos()
    preProcess.doc2vec()

    # PROCESO DE CLUSTERING

    algoritmo = DensityAlgorithm(preProcess.documentVectors, epsilon=5, minPt=10)
    algoritmo.ejectuarAlgoritmo()

    print(algoritmo.clusters)
