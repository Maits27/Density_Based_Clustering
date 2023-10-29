import numpy as np
from tqdm import tqdm

from sklearn.cluster import DBSCAN

from sklearn.metrics.pairwise import cosine_similarity

from scipy import spatial

from loadSaveData import saveDistances, loadDistances, saveClusters

def distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)


class DensityAlgorithmUrruela:

    def __init__(self, vectors, epsilon, minPt, dist={}):
        self.vectors = vectors  # DOCUMENTOS VECTORIZADOS
        self.epsilon = epsilon  # RADIO PARA CONSIDERAR VECINOS
        self.minPt = minPt  # MINIMO DE VECINOS PARA CONSIDERAR NUCLEO

        self.vecinos = []
        self.nucleos = []  # CONJUNTO DE INSTANCIAS NUCLEO
        self.clusters = [-1] * len(self.vectors) # CONJUNTO DE CLUSTERS
        self.distancias = dist # DISTANCIAS ENTRE VECTORES {frozenSet: float}
        self.clustersValidos = []  # CONJUNTO DE CLUSTERS SELECCIONADOS
        self.alcanzables = []
        self.dimensiones = len(vectors[0])
        self.typeDistance = 0 # 0 for Euclidean distance, 1 for cosine distance

    def ejecutarAlgoritmo(self):

        print('Ejecutando algoritmo:')
        self.calcular_distancias()

        self.buscarNucleos()
        self.crearClusters()
        self.seleccionClusters()

        self.reclasificarInst()
        self.reasignarLabelCluster()
        saveClusters(self.clusters, 'urruela')


    def calcular_distancias(self):
        if len(self.distancias) == 0:
            if (d:=loadDistances(nInstances=len(self.vectors), dimensions=self.dimensiones, typeDistance=self.typeDistance)) == False:
                for i, doc in tqdm(enumerate(self.vectors), desc=f'\tCALCULANDO DISTANCIAS, total de {len(self.vectors)}'):
                    for j, doc2 in enumerate(self.vectors):
                        if j != i:
                            if (pair := '_'.join(sorted([str(i), str(j)]))) not in self.distancias:
                                if self.typeDistance == 0:
                                    dist = np.linalg.norm(doc - doc2)
                                else:
                                    dist = 1 - spatial.distance.cosine(doc, doc2)
                                self.distancias[pair] = float(dist)
                print(f'\tTOTAL DE {len(self.distancias)} DISTANCIAS CALCULADAS')
                saveDistances(self.distancias, nInstances=len(self.vectors), dimensiones=self.dimensiones, typeDistance=self.typeDistance)
            else:
                print('\tDistancias encontradas y cargadas desde archivo')
                self.distancias = d


    def buscarNucleos(self):
        for i, _ in tqdm(enumerate(self.vectors), desc=f'\tBUSCANDO NUCLEOS, total de {len(self.vectors)}'):
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

    def getNumClusters(self):
        return len(set(self.clusters) - {-1})

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
        saveClusters(self.clusters, 'algorithm1')

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
        for i, doc in tqdm(enumerate(self.vectors), 'Ejecutando Maitane'):
            for j, doc2 in enumerate(self.vectors):
                if j != i:
                    if (pair := '_'.join(sorted([str(i), str(j)]))) not in self.distancias:
                        distEuc = 1-cosine_similarity([doc], [doc2])
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
        for i in tqdm(range(len(self.vectors)), 'Ejecutando Nagore'):
            if labels[i] == 0:
                neighbors = self.get_neighbors(self.vectors[i])
                if len(neighbors) < self.minPt:
                    labels[i] = -1
                else:
                    cluster_id = cluster_id + 1
                    self.expand_cluster(labels, i, neighbors, cluster_id)
        self.clusters = labels
        saveClusters(self.clusters, 'algorithm2')
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
        dbscan = DBSCAN(eps=self.epsilon, min_samples=self.minPt, metric='cosine')  # Ajusta los parámetros según tu caso
        self.clusters = dbscan.fit_predict(self.vectors)
        saveClusters(self.clusters, 'dbscan')


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

