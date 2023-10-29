import numpy as np
import pandas as pd
from pathlib import Path
import csv
import json


def loadRAWwithClass(path):
    """
    Devuelve un dataset con texto y clase
    """
    return pd.read_csv(path)


def loadRAW(path):
    """
    Devuelve un dataset con solo los textos
    """
    data = pd.read_csv(path)
    return [instancia[1] for instancia in data.values]
        

def formatoParaEmbeddingProjector(dim, l):
    file_name = f"../out/eProjectorTSV/VectoresDoc_L{l}_D{dim}.tsv"
    document_vectors = loadEmbeddings(length=l, dimension=dim)
    with open(file_name, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        for vector in document_vectors:
            writer.writerow(vector)


def saveTokens(textosTokenizados):
    length = len(textosTokenizados)
    ruta = Path(f'../out/tokens/tokens{length}.tok')
    if not ruta.exists():
        with open(ruta, "w", encoding="utf-8") as file:
            for texto in textosTokenizados:
                file.write('####\n')
                for token in texto:
                    file.write(token + "\n")


def loadTokens(length):
    path = Path(f'../out/tokens/tokens{length}.tok')
    if path.exists():
        textosTokenizados = []

        with open(path, 'r') as file:
            textoActual = []
            for line in file:
                if line == '####\n':
                    textoActualAux = textoActual.copy()
                    textosTokenizados.append(textoActualAux)
                    textoActual.clear()
                else:
                    textoActual.append(line.replace('\n',''))
        return textosTokenizados
    else:
        return False


def saveSinLimpiarTokens(textosTokenizados):
    length = len(textosTokenizados)
    ruta = Path(f'../out/tokens/tokens_sinlimpiar{length}.tok')
    if not ruta.exists():
        with open(ruta, "w", encoding="utf-8") as file:
            for texto in textosTokenizados:
                file.write('####\n')
                for token in texto:
                    file.write(token + "\n")


def loadSinLimpiarTokens(length):
    print('Cargando sin limpiar tokens')
    path = Path(f'../out/tokens/tokens_sinlimpiar{length}.tok')
    if path.exists():

        textosTokenizados = []

        with open(f'../out/tokens/tokens_sinlimpiar{length}.tok', 'r', encoding='utf-8') as file:
            textoActual = []
            for line in file:
                if line == '####\n':
                    textoActualAux = textoActual.copy()
                    textosTokenizados.append(textoActualAux)
                    textoActual.clear()
                else:
                    textoActual.append(line.replace('\n',''))
        return textosTokenizados
    else:
        return False


def saveEmbeddings(textEmbeddings, dimension, type='no-bert'):
    print('Guardando embeddings...')
    length = len(textEmbeddings)
    print("aaaaaaa",length)
    if type == 'bert': ruta = Path(f'../out/embeddings/bert/embeddings{length}dim{dimension}.npy')
    else: ruta = Path(f'../out/embeddings/embeddings{length}dim{dimension}.npy')
    if not ruta.exists(): # Only if file not exists
        np.save(ruta, textEmbeddings)


def loadEmbeddings(length, dimension=768, type='no-bert'):
    print('Cargando embeddings...')
    if type == 'bert':
        path = Path(f'../out/embeddings/bert/embeddings{length}dim{768}.npy')
        if path.exists(): loadedData = np.load(path)
        else: return False
    else: 
        path = Path(f'../out/embeddings/embeddings{length}dim{dimension}.npy')
        if path.exists(): loadedData = np.load(path)
        else: return False
    print('Embeddings cargados')
    return loadedData



def saveInCSV(nInstances, dimension, espilon, minPts, nClusters, silhouette):
    with open(f'../out/Barridos/TRANSFORMERSBarridos_D{dimension}_Epsilon{espilon}.csv', 'a') as file:
        writer = csv.writer(file, delimiter='|')
        writer.writerow([nInstances, dimension, espilon, minPts, nClusters, silhouette])


def saveInCSV2(nInstances, dimension, espilon, minPts, media_puntos_cluster, minimo_instancias,nClusters, silhouette):
    with open(f'../out/Barridos/TRANSFORMERSBarridos_D{dimension}_Epsilon{espilon}.csv', 'w', encoding='utf8') as file:
        file.write('N_Instances\tDim\tEps\tminPts\tmediaPuntosCluster\tminimoInstanciaCluster\tnClusters\tMetric\n')
        file.write(f'{nInstances}\t{dimension}\t{espilon}\t{minPts}\t{media_puntos_cluster}\t{minimo_instancias}\t{nClusters}\t{silhouette}')


def saveClusters(clusters, name):
    np.save(f'../out/cluster_labels/clusters_{name}', clusters)


def loadClusters(name):
    return np.load(f'../out/cluster_labels/clusters_{name}.npy')


def saveDistances(distancesDict, nInstances, dimensiones):
    ruta = Path(f'../out/distances/distances{nInstances}_dim{dimensiones}.json')
    if not ruta.exists():
        with open(f'../out/distances/distances{nInstances}_dim{dimensiones}.json', "w", encoding="utf-8") as archivo:
            json.dump(distancesDict, archivo, ensure_ascii=False)
        print('Distancias guardadas en JSON')


def loadDistances(nInstances, dimensions):
    print('Cargando distancias')
    path = Path(f'../out/distances/distances{nInstances}_dim{dimensions}.json')
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    else:
        return False
    