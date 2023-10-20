import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

def crearMatrizClassToCluster(data, clusters):
    clusters_validos = set(clusters)
    clases = data['class'].copy()
    print(len(clases))
    class_to_cluster = []
    for i in range(len(clusters_validos)):
        class_to_cluster.append([0, 0])
    for i, c in enumerate(clusters):
        if c != -1:
            if np.array(clases)[i].__eq__('suicide'):
                class_to_cluster[c][0] += 1
            else:
                class_to_cluster[c][1] += 1
    return class_to_cluster

def classToCluster(data, clusters):
    nombres_clases = ['suicide', 'non-suicide']
    class_to_cluster = crearMatrizClassToCluster(data, clusters)
    # Supongamos que tienes 20 grupos y 2 clases
    num_groups = len(set(clusters))
    num_classes = len(nombres_clases)
    # Crear una matriz de ejemplo con las instancias de clases en cada grupo
    # Esto es solo un ejemplo, debes proporcionar tus datos reales

    # Definir colores personalizados (azul y verde claros)
    light_blue = (0.6, 0.8, 1.0)  # Color azul claro
    light_green = (0.6, 1.0, 0.6)  # Color verde claro

    # Crear una figura y mostrar la matriz con los colores personalizados
    plt.figure(figsize=(10, 6))
    plt.imshow(class_to_cluster, cmap='viridis', aspect='auto', interpolation='nearest', vmin=0, vmax=10)

    # Personalizar el eje x y el eje y para mostrar los grupos y las clases
    plt.xticks(range(num_classes), [f'Clase {nombres_clases[i]}' for i in range(num_classes)])
    plt.yticks(range(num_groups), [f'{clusters[i]}' for i in range(num_groups)])

    # Usar los colores personalizados para mostrar la matriz
    for i in range(num_groups):
        for j in range(num_classes):
            plt.text(j, i, class_to_cluster[i][j], ha='center', va='center',
                     color=light_blue if j == 0 else light_green)

    # Etiquetas para los ejes
    plt.xlabel("Clases")
    plt.ylabel("Grupos")

    plt.title("Matriz de Clases Asignadas a Grupos")
    plt.show()


def wordCloud(clusters, textos_tokenizados):
    palabras_del_cluster = []
    for c in range(-1, max(clusters) + 1):
        palabras_c = ' '
        for i, clus in enumerate(clusters):
            if clus == c:
                t = ' '.join(textos_tokenizados[i])
                palabras_c = palabras_c + ' ' + t
        palabras_del_cluster.append(palabras_c)

    for i in range(len(palabras_del_cluster)):
        print(
            f'\n\n\n###################################CLUSTER {i - 1}##########################################\n\n\n')
        wc = WordCloud(width=300, height=300).generate(palabras_del_cluster[i])
        plt.axis("off")
        plt.imshow(wc, interpolation="bilinear")
        plt.show()
