import numpy as np

def saveTokens(textosTokenizados):
    # TODO faltan los ####. NO FUNCIONA
    length = len(textosTokenizados)

    with open(f'../out/tokens/tokens{length}.tok', "w", encoding="utf-8") as file:
        for token in textosTokenizados:
            file.write(token.text + "\n")


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
    print('Guardando embeddings...')
    length = len(textEmbeddings)
    np.save(f'../out/embeddings/embeddings{length}dim{dimension}.npy', textEmbeddings)


def loadEmbeddings(length, dimension):
    print('Cargando embeddings...')
    return np.load(f'../out/embeddings/embeddings{length}dim{dimension}.npy')
    