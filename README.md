[https://radimrehurek.com/gensim/models/doc2vec.html](https://radimrehurek.com/gensim/models/doc2vec.html)

[datasets](https://drive.google.com/drive/folders/10w1BTdpTPpNzsvfYKhr-Y2R-h7G4SuJn?usp=drive_link)

[seguimiento de horas](https://docs.google.com/spreadsheets/d/1L2ZBiSUnDmHg6R7CX__1veOn-hM27sVQc-pTQLwuiZo/edit?usp=drive_link)

[documentación Overleaf](https://www.overleaf.com/project/652946916091309cffe6d295)

- [1. Introducción](#1-introducción)
- [2. Estructura del proyecto](#2-estructura-del-proyecto)
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

La elección del tamaño del dataset se ha hecho en base a los siguientes tiempos:


|                 | Limpieza, tokenización y lematización |  doc2vec (150 dim) | sklearn.DBSCAN (epsilon:2, minPts: 2) |
|-----------------|---------------------------------------|--------------------|---------------------------------------|
|100 instances:   |                   3.2s                |          0.9s      |                   0.5s                |
|1000 instances:  |                  23.4s                |          5.8s      |                   3.7s                |
|5000 instances:  |               1m 55.7s                |         29.6s      |                  19.9s                |
|10000 instances: |               3m 56.9s                |       1m 0.9s      |        47s                |
|20000 instances: |               7m 47.1s                |       2m 0.0s      |               1m 29.4s                |

# 2. Estructura del proyecto

## 2.1. Proceso prinicipal

El proceso principal está dividido en diferentes módulos que siguen el siguiente diagrama:

<img src="img/diagrams/mainProcess.jpg" alt="drawing" width="500"/>

### 2.1.1. reduceDataset.py

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



