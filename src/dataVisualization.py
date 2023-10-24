import sys

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from clustering import DensityAlgorithm, DBScanOriginal
import numpy as np
from loadSaveData import loadEmbeddings
import csv
import optuna
import plotly.express as px

def distance_distribution(nInstances, dimension):
    pares_calculados = set()
    distancias = []
    ema = [0] * 10
    embeddingVectors = loadEmbeddings(length=nInstances, dimension=dimension)
    for i, doc in enumerate(embeddingVectors):
        for j, doc2 in enumerate(embeddingVectors):
            if j != i:
                if (pair := '_'.join(sorted([str(i), str(j)]))) not in pares_calculados:
                    distancias.append(np.linalg.norm(doc - doc2))
                    pares_calculados.add(pair)
    for dist in distancias:
        index = int(dist/5)
        if index>10: ema[9] = ema[9]+1
        else: ema[index] = ema[index]+1
    print(f'LOS TARTES DE LAS DISTANCIAS: {ema}')
    fig = px.histogram(x=distancias, nbins=20)
    fig.show()


def PCA():
    pass


def tSNE():
    pass

#distance_distribution(nInstances=10000, dimension=250)