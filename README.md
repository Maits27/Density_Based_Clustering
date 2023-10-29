- [1. Introducción](#1-introducción)
  - [1.1. Dataset](#11-dataset)
  - [1.2. Ejecución](#12-ejecución)
- [2. Estructura del proyecto - Código fuente](#2-estructura-del-proyecto---código-fuente)
  - [2.1. Proceso prinicipal](#21-proceso-prinicipal)
    - [2.1.1. reduceDataset.py](#211-reducedatasetpy)
    - [2.1.2. tokenization.py](#212-tokenizationpy)
    - [2.1.3. vectorization.py](#213-vectorizationpy)
    - [2.1.4. clustering.py](#214-clusteringpy)
    - [2.1.5. evaluation.py](#215-evaluationpy)
  - [2.2. Proceso de inferencia](#22-proceso-de-inferencia)
  - [2.3. inference.py](#23-inferencepy)
  - [2.4. Módulos de utilidades](#24-módulos-de-utilidades)
    - [2.4.1. paramOptimization.py](#241-paramoptimizationpy)
    - [2.4.2. dataVisualization.py](#242-datavisualizationpy)
    - [2.4.3. loadSaveData.py](#243-loadsavedatapy)


# 1. Introducción

## 1.1. Dataset

El dataset se puede obtener de [kaggle.com/datasets/nikhileswarkomati/suicide-watch](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch). Este dataset tiene que ser añadido a la carpeta `./Datasets`

La elección del tamaño del dataset se puede hacer teniendo en cuenta los siguientes tiempos:


|                 | Limpieza, tokenización y lematización |  doc2vec (150 dim) | sklearn.DBSCAN (epsilon:2, minPts: 2) |
|-----------------|---------------------------------------|--------------------|---------------------------------------|
|100 instances:   |                   3.2s                |          0.9s      |                   0.5s                |
|1000 instances:  |                  23.4s                |          5.8s      |                   3.7s                |
|5000 instances:  |               1m 55.7s                |         29.6s      |                  19.9s                |
|10000 instances: |               3m 56.9s                |       1m 0.9s      |                   47s                 |
|20000 instances: |               7m 47.1s                |       2m 0.0s      |               1m 29.4s                |


## 1.2. Ejecución

Requisitos:
* Tener `python3` instalado
* Tener `pip` instalado

1. Clonar el repositorio:

```bash
git clone https://github.com/Maits27/PruebitasMineria.git
```

2. Instalar las dependencias

```bash
cd ./PruebitasMineria
pip install -r requirements.txt
```

3. Descargar dataset de [kaggle.com/datasets/nikhileswarkomati/suicide-watch](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch) y añadirlo a la carpeta `./Datasets` del proyecto.

4. Ejecutar el archivo `./src/main.py`:

```bash
cd ./src/
python main.py <numInstances> <vectorsDim> <vectorType> <algorithm> <epsilon> <minPts>
```

Donde:
* `numInstances`: número de instancias a utilizar de entrenamiento. Tiene que ser menor al número de instancias totales del dataset original.
* `vectorsDim`: número de dimensiones a utilizar. Indicar cualquier número si se usa `bert` en `vectorType`
* `vectorType`: `bert` para usar Transformers, `doc2vec` para usar Doc2Vec o `tfidf` para usar `TF-IDF`
* `algorithm`: `ourDensityAlgorithm` para usar nuestro algoritmo y `dbscan` para usar el de la librería `sklearn`.
* `epsilon`: valor de `epsilon` para el algortimo de Clustering.
* `minPts`: valor de `minPts` para el algoritmo de Clustering.

# 2. Estructura del proyecto - Código fuente

## 2.1. Proceso prinicipal

El proceso principal está dividido en diferentes módulos que siguen el siguiente diagrama:

<img src="img/diagrams/mainProcess.jpg" alt="drawing" width="500"/>

### 2.1.1. reduceDataset.py

Modulo para reducir el dataset original. Se pueden indicar los valores para un dataset `train` y un datset `test`:

```bash
python reduceDatset.py <datasetPath> <numTrainInstances> <numTestInstances> <outPath> 
```

Donde:
* `datasetPath`: ruta del dataset original. Preferiblemente en `./Datasets`.
* `numTrainInstances`: número de instancias para `train`.
* `numTestInstances`: número de instancias para `test`. La suma del número de instancias para `train` y el número de instancias para `test` no debe superar el número de instancias del dataset original.
* `outPath`: ruta para escribir los dataset `train` y `test`. Preferiblemente en `./Datasets`.

### 2.1.2. tokenization.py

### 2.1.3. vectorization.py

### 2.1.4. clustering.py

### 2.1.5. evaluation.py

## 2.2. Proceso de inferencia

El de inferencia utiliza los módulos del siguiente diagrama:

<img src="img/diagrams/inferenceProcess.jpg" alt="drawing" width="500"/>

## 2.3. inference.py

## 2.4. Módulos de utilidades

### 2.4.1. paramOptimization.py

### 2.4.2. dataVisualization.py

### 2.4.3. loadSaveData.py



