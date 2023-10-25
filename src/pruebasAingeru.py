from loadSaveData import loadEmbeddings, loadRAW
from clustering import DBScanOriginal
from tokenization import tokenizarSinLimpiar
from evaluation import wordCloud


vectors = loadEmbeddings(length=10000, dimension=768, type='bert')
algoritmo = DBScanOriginal(vectors=vectors, epsilon=0.007, minPt=9)
algoritmo.ejecutarAlgoritmo()

rawData = loadRAW('../Datasets/Suicide_Detection10000.csv')
tokenTexts = tokenizarSinLimpiar(rawData)

wordCloud(algoritmo.clusters, textos_tokenizados=tokenTexts)