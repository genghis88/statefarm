import pandas as pd
from os import path
import pickle
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxoutDense, Activation, Convolution1D, MaxPooling1D, Flatten, RepeatVector, AveragePooling1D
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam, Adamax
from keras.layers.advanced_activations import ELU, PReLU

trainFile = sys.argv[1]
pickleFile = sys.argv[2]

trainingDir = 'data/train'
testDir = 'data/test'

trainingLabels = pd.read_csv(trainFile)

#print(trainingLabels[trainingLabels['subject'] == 'p002'])

imageSize = 921600
xTrain = np.zeros((trainingLabels.shape[0], imageSize), dtype='float32')
for index, row in trainingLabels.iterrows():
  fileName = path.join(trainingDir, row['classname'], row['img'])
  print(fileName)