import pandas as pd

datsetPath = '../Datasets/Suicide_Detection.csv'

data = pd.read_csv(datsetPath)
print('Original datset length', len(data))

numberOfInstancesToGet = 10000
reducedDataset = data.head(n=numberOfInstancesToGet)

pathToWirte = f'../Datasets/Suicide_Detection{numberOfInstancesToGet}.csv'
reducedDataset.to_csv(pathToWirte, index=False)

# Check
dataCheck = pd.read_csv(pathToWirte)
print('Reduced datset length', len(dataCheck))