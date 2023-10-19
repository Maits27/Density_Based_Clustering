import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from main import DensityAlgorithm
import numpy as np


# TODO esto habrá que cogerlo de algún lado, no así hardcodeado
dimensions = 150
textos_tokenizados = []
texto_embedding = []

def heuristicoEpsilonDBSCAN():
    nn = NearestNeighbors(n_neighbors=dimensions*2)
    nn.fit(texto_embedding)
    distances, idx = nn.kneighbors(texto_embedding)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    plt.show()

def barridoDBSCAN(espilonList, minPtsList):
    DensityAlgorithm(vectors=texto_embedding, epsilon=1, minPt=1)

def cargarTokens():
    with open('../tokens.tok', 'r') as file:
        for line in file:
            # TODO
            print(line)

cargarTokens()