import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud
import matplotlib.colors as mcolors

'''def crearMatrizClassToCluster(data, clusters):
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
    return class_to_cluster'''
def pairWiseEvaluation(clusters1, clusters2):
    '''
    :param clusters1: clusters del primer metodo
    :param clusters2: clusters del segundo metodo
    :return: la matriz arrayComparativos (pair-wise comparison), donde:
        0.Pos: coinciden en ambos metodos
        1.Pos: coinciden en el primero
        2.Pos: coinciden en el segundo
        3.Pos: no coinciden
    '''
    arrayComparativo = [0]*4
    paresComparados = set()
    for index1, c1 in enumerate(clusters1):
        for index2, c2 in enumerate(clusters1):
            if index2 != index1:
                if (pair := '_'.join(sorted([str(index1), str(index2)]))) not in paresComparados:
                    paresComparados.add(pair)
                    if c1 == c2 and clusters2[index1] == clusters2[index2]:
                        arrayComparativo[0] += 1
                    elif c1 == c2 and not clusters2[index1] == clusters2[index2]:
                        arrayComparativo[1] += 1
                    elif not c1 == c2 and clusters2[index1] == clusters2[index2]:
                        arrayComparativo[2] += 1
                    else:
                        arrayComparativo[3] += 1
    print(arrayComparativo)
    return arrayComparativo

def clase_a_num(data):
    clases = data['class'].copy()
    res = []
    for i, c in enumerate(clases):
        if c.__eq__('suicide'):
            res.append(-1)
        else:
            res.append(0)
    return res


def classToCluster(data, clusters):
    """
    data: tiene que ser un dataframe con un campo 'text' y 'class'
    """
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
    plt.yticks(np.arange(num_groups), [f' {i}' for i in range(min(clusters), max(clusters) + 1)])
    thresh = cm.max() / 2.

    for i in range(len(set(clusters))):
        for j in range(num_classes):
            plt.text(j, i, format(cm[i][j], 'd'), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    # Etiquetas para los ejes
    plt.xlabel("Class")
    plt.ylabel("Cluster")

    plt.title("Class2Cluster Matrix")
    plt.savefig(f'../img/tmp/matrix')
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
        plt.savefig(f'../img/tmp/wordCloud{i - 1}')
        plt.show()

    
def getClusterSample(clusterList, numClusters, rawData, sample=10):
    foundInstancesForEachCluster = {}
    for index, instance in enumerate(clusterList):
        if instance not in foundInstancesForEachCluster:
            foundInstancesForEachCluster[instance] = [rawData[index]]
        elif len(foundInstancesForEachCluster[instance]) != sample:
            foundInstancesForEachCluster[instance].append(rawData[index])
        if _areEnoughSamples(foundInstancesForEachCluster, numClusters,sample):
            _printSamples(foundInstancesForEachCluster)
            return foundInstancesForEachCluster
    _printSamples(foundInstancesForEachCluster)
    return foundInstancesForEachCluster


def _areEnoughSamples(dict, numClusters,sample):
    if len(dict.keys()) == numClusters:
        for key in dict.keys():
            if len(dict[key]) < sample:
                return False
        return True
    else:
        return False        
    

def _printSamples(samplesDict):
    for key in samplesDict.keys():
        print(f'### Cluster {key} ###')
        for index, text in enumerate(samplesDict[key]):
            print(f'\t### Instance {index} ###')
            print(f'\t{text}')
            print('\n\n')
        print('\n\n\n')