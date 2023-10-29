import pandas as pd
import numpy as np
from loadSaveData import loadRAW
import sys


def isPossibleToSplit(totalInstances, numTrainInstances, numTestInstances):
	return (numTrainInstances + numTestInstances) <= totalInstances


def reduceDataset(path, numTrainInstances, numTestInstances):
	data = loadRAW(path)

	if isPossibleToSplit(len(data), numTrainInstances, numTestInstances):
		trainDataset = data.head(n=numTrainInstances)
		testDataset = data.tail(n=numTestInstances)

		trainDataset.to_csv(f'../Datasets/Suicide_Detection_train{numTrainInstances}(test{numTestInstances}).csv', index=False)
		testDataset.to_csv(pathToWrite + f'../Datasets/Suicide_Detection_test{numTestInstances}(train{numTrainInstances}).csv', index=False)

		return trainDataset, testDataset


if __name__ == '__main__':
	datsetPath = sys.argv[1]
	numTrainInstances = int(sys.argv[2])
	numTestInstances = int(sys.argv[3])
	pathToWrite = sys.argv[4]

	reduceDataset(datsetPath, numTrainInstances, numTestInstances)
