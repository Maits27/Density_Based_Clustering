import pandas as pd
import spacy
import emoji
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import tqdm
import time
import json
from pathlib import Path

from sklearn.cluster import DBSCAN

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import get_tmpfile
from sklearn.metrics.pairwise import cosine_similarity

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

    def tfidf(self):
        vectorizer = TfidfVectorizer()
        self.documentVectors = vectorizer.fit_transform(self.textos)
        saveEmbeddings(self.documentVectors, self.dimensiones)
    def doc2vec(self):

        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(self.textos_token)]
        model = Doc2Vec(documents, vector_size=self.dimensiones, window=2, dm=1, epochs=100, workers=4)

        model.build_vocab(documents)
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

        model.save(get_tmpfile("my_doc2vec_model"))

        self.documentVectors = [model.infer_vector(doc) for doc in self.textos_token]

        saveEmbeddings(self.documentVectors, self.dimensiones)


class DensityAlgorithm:

    def __init__(self, vectors, epsilon, minPt, dim):
        self.vectors = vectors  # DOCUMENTOS VECTORIZADOS
        self.epsilon = epsilon  # RADIO PARA CONSIDERAR VECINOS
        self.minPt = minPt  # MINIMO DE VECINOS PARA CONSIDERAR NUCLEO

        self.vecinos = []
        self.nucleos = []  # CONJUNTO DE INSTANCIAS NUCLEO
        self.clusters = [-1] * len(self.vectors) # CONJUNTO DE CLUSTERS
        self.distancias = {} # DISTANCIAS ENTRE VECTORES {frozenSet: float}
        self.clustersValidos = []  # CONJUNTO DE CLUSTERS SELECCIONADOS
        self.alcanzables = []
        self.dimensiones = dim

    def ejectuarAlgoritmo(self):
        archivoPath = Path(f'distanciasEuc{self.dimensiones}')
        if not archivoPath.exists():
            self.calcular_distancias()
        else:
            with open(archivoPath, "r") as archivo:
                self.distancias = json.load(archivo)
        self.buscarNucleos()
        self.crearClusters()
        self.seleccionClusters()

        self.reclasificarInst()
        self.reasignarLabelCluster()

    def calcular_distancias(self):
        print('CALCULANDO DISTANCIAS')
        '''archivoPath = Path(f'distanciasEuc{self.dimensiones}')
        if not archivoPath.exists():'''
        for i, doc in enumerate(self.vectors):
            for j, doc2 in enumerate(self.vectors):
                if j != i:
                    if (pair := '_'.join(sorted([str(i), str(j)]))) not in self.distancias:
                        distEuc = np.linalg.norm(doc - doc2) # 1-cosine_similarity([doc], [doc2])
                        self.distancias[pair] = distEuc
        print(f'TOTAL DE {len(self.distancias)} DISTANCIAS CALCULADAS')

            # with open(archivoPath, "w", encoding="utf-8") as archivo:
            #     # Utilizar json.dump() para escribir el diccionario en el archivo
            #     json.dump(self.distancias, archivo, ensure_ascii=False)

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
        for i in tqdm(self.nucleos, desc='EXPLORANDO NUCLEOS'):
            if self.clusters[i] == -1:
                numCluster += 1
                self.clusters[i] = numCluster
                nucleosPorVisitar.append(i)

                while nucleosPorVisitar:
                    j = nucleosPorVisitar.pop()

                    for index in self.vecinos[j]:
                        if self.clusters[index] == -1:
                            self.clusters[index] = numCluster
                            '''if index in self.nucleos:
                                nucleosPorVisitar.append(index)'''

    def seleccionClusters(self):
        for c in range(len(set(self.clusters))):
            if self.clusters.count(c) < self.minPt:
                for i, clus in enumerate(self.clusters):
                    if clus == c:
                        self.alcanzables.append((i, self.vecinos[i]))
            else:
                self.clustersValidos.append(c)

    def reclasificarInst(self):
        for i, vecinos in self.alcanzables:
            previo = self.clusters[i]
            for j in vecinos:
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
        for cluster in range(min(self.clusters), max(self.clusters) + 2):
            kont = 0
            for i in self.clusters:
                if i == cluster:
                    kont += 1
            if cluster == -1:
                print(f'Hay un total de {kont} instancias que son ruido')
            else:
                print(f'Del cluster {cluster} hay {kont} instancias')

def llamar_al_metodo(preProcess, epsilon, minPt):

    # CLUSTERING ALTERNATIVO
    start_time = time.time()
    algoritmo = DensityAlgorithm(preProcess.documentVectors, epsilon=epsilon, minPt=minPt, dim=preProcess.dimensiones)
    algoritmo.ejectuarAlgoritmo()
    end_time = time.time()

    print(f'\n\n\nTiempo Maitane: {end_time - start_time}')
    algoritmo.imprimir()

    classToCluster(preProcess.data, algoritmo.clusters)
    wordCloud(algoritmo.clusters, preProcess.textos_token)




if __name__ == '__main__':
    start_time = time.time()
    # PREPROCESADO DE DATOS
    preProcess = PreProcessing('../Datasets/Suicide_Detection10000.csv', 150)
    preProcess.cargarDatos()
    # PREPROCESO SIN HACER
    preProcess.limpiezaDatos()
    #preProcess.doc2vec()

    # PREPROCESADO HECHO:
    preProcess.documentVectors = loadEmbeddings(10000, 150)
    #preProcess.textos_token = loadTokens(10000)
    end_time = time.time()
    print(f'\n\n\nTiempo preproceso: {end_time - start_time}')

    # PROCESO DE CLUSTERING
    # PARAMETROS:
    epsilon = 12.25#15#10
    minPt = 10#10#5
    llamar_al_metodo(preProcess, epsilon, minPt) # MAITANE
