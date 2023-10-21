import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud
import matplotlib.colors as mcolors

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

def clase_a_num(data):
    clases = data['class'].copy()
    res = []
    for c in clases:
        if c.__eq__('suicide'):
            res.append(0)
        else:
            res.append(1)
    return res
def classToCluster(data, clusters):
    cm = confusion_matrix(clusters, clase_a_num(data))
    # Supongamos que tienes 20 grupos y 2 clases
    num_groups = len(set(clusters))
    num_classes = len(nombres_clases := ['suicide', 'non-suicide'])

    if num_classes < num_groups:
        cm = cm[:, :num_classes]
    elif num_groups < num_classes:
        cm = cm[:num_classes, :]

    plt.figure(figsize=(10, 6))
    plt.imshow(cm, cmap=plt.cm.Blues, aspect='auto', interpolation='nearest', vmin=0, vmax=10)

    # Personalizar el eje x y el eje y para mostrar los grupos y las clases
    plt.xticks(np.arange(num_classes), [f'Class {nombres_clases[i]}' for i in range(num_classes)])
    plt.yticks(np.arange(num_groups), np.unique(clusters))
    thresh = cm.max() / 2.

    for i in range(num_groups):
        for j in range(num_classes):
            plt.text(j, i, format(cm[i][j], 'd'), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    # Etiquetas para los ejes
    plt.xlabel("Class")
    plt.ylabel("Cluster")

    plt.title("Class2Cluster Matrix")
    plt.show()


def wordCloud(clusters, textos_tokenizados):
    palabras_del_cluster = []
    for c in range(min(clusters), max(clusters) + 1):
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
