import pandas as pd
import numpy as np
datsetPath = '../Datasets/Suicide_Detection.csv'

data = pd.read_csv(datsetPath)
print('Original datset length', len(data))

numberOfInstancesToGet = 10000

reducedDataset = data.head(n=numberOfInstancesToGet)

pathToWirte = f'../Datasets/Suicide_Detection{numberOfInstancesToGet}.csv'
reducedDataset.to_csv(pathToWirte, index=False)

# Check
dataCheck = pd.read_csv(pathToWirte)
print('Reduced dataset length', len(dataCheck))




# GENERATE TEST.CSV
datsetPath = '../Datasets/Suicide_Detection.csv'

data = pd.read_csv(datsetPath)
print('Original datset length', len(data))

numberOfInstancesToGet = 100

reducedDataset = data.tail(n=numberOfInstancesToGet)

pathToWrite = f'../Datasets/Suicide_Detection{numberOfInstancesToGet}_test.csv'
reducedDataset.to_csv(pathToWrite, index=False)

# Check
dataCheck = pd.read_csv(pathToWrite)
print('Reduced test dataset length', len(dataCheck))
