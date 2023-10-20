import numpy as np
from pathlib import Path


def saveTokens(textosTokenizados):
    # TODO faltan los ####. NO FUNCIONA
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


def saveEmbeddings(textEmbeddings, dimension):
    print('Guardando embeddings...') # TODO lo he cambiado para que no lo haga siempre (solo si no esta creado ya)
    length = len(textEmbeddings)
    ruta = Path(f'../out/embeddings/embeddings{length}dim{dimension}.npy')
    if not ruta.exists():
        np.save(ruta, textEmbeddings)



def loadEmbeddings(length, dimension):
    print('Cargando embeddings...')
    return np.load(f'../out/embeddings/embeddings{length}dim{dimension}.npy')
    