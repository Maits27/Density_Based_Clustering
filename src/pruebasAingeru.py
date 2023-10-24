from main import DensityAlgorithm, DensityAlgorithm2, DBScanOriginal
from loadSaveData import loadEmbeddings
import time
from sklearn.metrics import silhouette_score

bert = loadEmbeddings(length=10000, dimension=768, type='bert')
noBert = loadEmbeddings(length=10000, dimension=250)

print(len(bert))
print(len(noBert))

print(bert[0])
print(noBert[0])