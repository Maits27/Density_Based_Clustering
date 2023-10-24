import numpy as np
import pandas as pd
from pathlib import Path
import csv


def loadRAW(path):
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
    textosTokenizados = []

    with open(f'../out/tokens/tokens{length}.tok', 'r') as file:
        textoActual = []
        for line in file:
            if line == '####\n':
                textoActualAux = textoActual.copy()
                textosTokenizados.append(textoActualAux)
                textoActual.clear()
            else:
                textoActual.append(line.replace('\n',''))

    return textosTokenizados


def saveEmbeddings(textEmbeddings, dimension, type='no-bert'):
    print('Guardando embeddings...')
    length = len(textEmbeddings)
    if type == 'bert': ruta = Path(f'../out/embeddings/bert/embeddings{length}dim{dimension}.npy')
    else: ruta = Path(f'../out/embeddings/embeddings{length}dim{dimension}.npy')
    if not ruta.exists(): # Only if file not exists
        np.save(ruta, textEmbeddings)


def loadEmbeddings(length, dimension, type='no-bert'):
    print('Cargando embeddings...')
    if type == 'bert': return np.load(f'../out/embeddings/bert/embeddings{length}dim{dimension}.npy')
    else: return np.load(f'../out/embeddings/embeddings{length}dim{dimension}.npy')


def saveInCSV(nInstances, dimension, espilon, minPts, nClusters, silhouette):
    with open(f'../out/Barridos/TRANSFORMERSBarridos_D{dimension}_Epsilon{espilon}.csv', 'a') as file:
        writer = csv.writer(file, delimiter='|')
        writer.writerow([nInstances, dimension, espilon, minPts, nClusters, silhouette])


def saveInCSV2(nInstances, dimension, espilon, minPts, media_puntos_cluster, minimo_instancias,nClusters, silhouette):
    with open(f'../out/Barridos/TRANSFORMERSBarridos_D{dimension}_Epsilon{espilon}.csv', 'w', encoding='utf8') as file:
        file.write('N_Instances\tDim\tEps\tminPts\tmediaPuntosCluster\tminimoInstanciaCluster\tnClusters\tMetric\n')
        file.write(f'{nInstances}\t{dimension}\t{espilon}\t{minPts}\t{media_puntos_cluster}\t{minimo_instancias}\t{nClusters}\t{silhouette}')

formatoParaEmbeddingProjector(250, 10000)