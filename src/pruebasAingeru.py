from main import DensityAlgorithm, DensityAlgorithm2, DBScanOriginal
from loadSaveData import loadEmbeddings
import time
from sklearn.metrics import silhouette_score

epsilon = 25
minPt = 1
documentVectors = loadEmbeddings(10000, 150)

"""
print('Ejecutando Maitane')
start_time = time.time()
algoritmo = DensityAlgorithm(documentVectors, epsilon=epsilon, minPt=minPt)
algoritmo.ejectuarAlgoritmo()
end_time = time.time()

print(f'\n\n\nTiempo Maitane: {end_time-start_time}')
algoritmo.imprimir()
"""

"""
# CLUSTERING DBSCAN IMPLEMENTADO
print('Ejecutando Nagore')
start_time = time.time()
algoritmo2 = DensityAlgorithm2(documentVectors, epsilon=epsilon, minPt=minPt)
algoritmo2.ejecutarAlgoritmo()
end_time = time.time()

print(f'\n\n\nTiempo Nagore: {end_time-start_time}')
algoritmo2.imprimir()
"""


# CLUSTERING DBSCAN ORIGINAL
print('Ejecutando Original')
start_time = time.time()
algoritmo3 = DBScanOriginal(documentVectors, epsilon=epsilon, minPt=minPt)
algoritmo3.ejecutarAlgoritmo()
end_time = time.time()

print(f'\n\n\nTiempo DBScan: {end_time - start_time}')
print(silhouette_score(documentVectors, algoritmo3.clusters))
algoritmo3.imprimir()