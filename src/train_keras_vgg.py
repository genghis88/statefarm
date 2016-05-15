import pandas as pd
from os import path
from os import listdir
from PIL import Image
import pickle
import sys
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, MaxoutDense, Activation, Convolution2D, MaxPooling2D, Flatten, AveragePooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam, Adamax
from keras.layers.advanced_activations import ELU, PReLU
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn import cross_validation
import gc

trainFile = sys.argv[1]
pickleFile = sys.argv[2]
predictionsFile = sys.argv[3]
source = sys.argv[4]

trainingDir = 'data/train_normalized'
testDir = 'data/test_normalized'
imageSize = 1200
imageShape = (40, 30)
batchSize = 256
numChannels = 1
gcEnabled = False
if source == '1':
  trainingDir = 'data/train_normalized'
  testDir = 'data/test_normalized'
  imageSize = 1200
  imageShape = (40, 30)
  batchSize = 4096
elif source == '2':
  trainingDir = 'data/train_normalized_big'
  testDir = 'data/test_normalized_big'
  imageSize = 19200
  imageShape = (160, 120)
  batchSize = 1024
elif source == '3':
  trainingDir = 'data/train_normalized_square_big'
  testDir = 'data/test_normalized_square_big'
  imageSize = 14400
  imageShape = (120, 120)
  batchSize = 8
  gcEnabled = True
elif source == '4':
  trainingDir = 'data/train_normalized_80_60'
  testDir = 'data/test_normalized_80_60'
  imageSize = 4800
  imageShape = (80, 60)
  batchSize = 128

trainingLabels = pd.read_csv(trainFile)
print(trainingLabels.shape)
#print(trainingLabels)

#originalY = np.array([int(x[-1:]) for x in trainingLabels['classname']]).astype('int32')
#y = to_categorical(originalY, 10)

#includes start, excludes end
def getTrainImages(start, end):
  xTrainBatch = []
  yTrainBatch = []
  for index in range(start,end):
    row = trainingLabels.loc[index]
    fileName = path.join(trainingDir, row['classname'] + row['img'])
    im = Image.open(fileName)
    xTrainBatch.append(np.reshape(im, imageSize))
    im.close()
    yTrainBatch.append(int(row['classname'][-1:]))
  xTrainBatch = np.array(xTrainBatch).astype('float32')
  xTrainBatch /= 255
  xTrainBatch = xTrainBatch.reshape((end - start), 1, imageShape[0], imageShape[1])
  yTrainBatch = np.array(yTrainBatch).astype('int32')
  yTrainBatch = to_categorical(yTrainBatch, 10)
  return xTrainBatch, yTrainBatch

files = [ f for f in listdir(testDir) if path.isfile(path.join(testDir,f)) ]
def getTestImages(start, end):
  xTrainBatch = []
  yTrainBatch = []
  for index in range(start,end):
    fName = files[index]
    fileName = path.join(testDir,fName)
    im = Image.open(fileName)
    xTestBatch.append(np.reshape(im, imageSize))
    im.close()
  xTestBatch = np.array(xTestBatch).astype('float32')
  xTestBatch /= 255
  xTestBatch = xTestBatch.reshape((end - start), 1, imageShape[0], imageShape[1])
  return xTestBatch

def getNextTrainBatch(batchFirstIndex):
  if batchFirstIndex + batchSize >= trainingLabels.shape[0]:
    return getTrainImages(batchFirstIndex, trainingLabels.shape[0] - 1)
  else:
    return getTrainImages(batchFirstIndex, batchFirstIndex + batchSize)

def getNextTestBatch(batchFirstIndex):
  if batchFirstIndex + batchSize >= len(files):
    return getTestImages(batchFirstIndex, len(files) - 1)
  else:
    return getTestImages(batchFirstIndex, batchFirstIndex + batchSize)

def getNet1():
  net = Sequential()
  net.add(Convolution2D(32, 3, 3, input_shape=(numChannels, imageShape[0], imageShape[1]), init='glorot_uniform'))
  net.add(ELU())
  net.add(MaxPooling2D(pool_size=(2,2)))
  net.add(Convolution2D(64, 4, 3, init='glorot_uniform'))
  net.add(Activation('tanh'))
  net.add(MaxPooling2D(pool_size=(2,2)))
  net.add(Convolution2D(128, 3, 3, init='glorot_uniform'))
  net.add(Activation('tanh'))
  net.add(MaxPooling2D(pool_size=(2,2)))
  net.add(Convolution2D(256, 3, 2, init='glorot_uniform'))
  net.add(ELU())
  net.add(Flatten())
  net.add(Dense(1024, init='glorot_uniform'))
  net.add(Activation('tanh'))
  net.add(Dropout(0.5))
  net.add(Dense(512, init='glorot_uniform'))
  net.add(Activation('tanh'))
  net.add(Dense(10, activation='softmax'))
  return net

def getNet2():
  net = Sequential()
  net.add(ZeroPadding2D((1,1), input_shape=(numChannels, imageShape[0], imageShape[1])))
  net.add(Convolution2D(32, 3, 3, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(32, 3, 3, init='glorot_uniform'))
  net.add(MaxPooling2D(pool_size=(2,2))) #20x15

  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(64, 3, 3, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(64, 3, 3, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(64, 5, 4, init='glorot_uniform')) #18x14
  net.add(Activation('relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #9x7

  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(128, 3, 3, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(128, 3, 3, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(128, 4, 4, init='glorot_uniform')) #8x6
  net.add(Activation('relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #4x3

  net.add(Convolution2D(256, 3, 2, init='glorot_uniform')) #2x2
  net.add(Activation('relu'))

  net.add(Flatten())

  net.add(Dense(1024, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(Dropout(0.5))
  net.add(Dense(512, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(Dropout(0.5))
  net.add(Dense(256, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(Dense(10, activation='softmax'))
  return net

def getNet3():
  net = Sequential()
  net.add(Convolution2D(32, 7, 7, input_shape=(xTrain.shape[1], xTrain.shape[2], xTrain.shape[3]), init='glorot_uniform')) # 154x114
  net.add(ELU())
  net.add(MaxPooling2D(pool_size=(2,2)))
  net.add(Dropout(0.5))
  net.add(Convolution2D(64, 6, 6, init='glorot_uniform')) # 72x52
  net.add(Activation('tanh'))
  net.add(MaxPooling2D(pool_size=(2,2)))
  net.add(Dropout(0.5))
  net.add(Convolution2D(128, 5, 5, init='glorot_uniform')) # 32x22
  net.add(Activation('tanh'))
  net.add(MaxPooling2D(pool_size=(2,2)))
  net.add(Dropout(0.5))
  net.add(Convolution2D(256, 5, 4, init='glorot_uniform')) # 12x8
  net.add(ELU())
  net.add(MaxPooling2D(pool_size=(2,2)))
  net.add(Dropout(0.5))
  net.add(Convolution2D(256, 3, 3, init='glorot_uniform')) # 4x2
  net.add(Activation('tanh'))
  net.add(Flatten())
  net.add(Dense(2048, init='glorot_uniform'))
  net.add(Activation('tanh'))
  net.add(Dense(1024, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(Dense(512, init='glorot_uniform'))
  net.add(Activation('tanh'))
  net.add(Dense(256, init='glorot_uniform'))
  net.add(Activation('tanh'))
  net.add(Dense(10, activation='softmax'))
  return net

if source == '1':
  #net = getNet1()
  net = getNet2()
  numberEpochs = 10
elif source == '2':
  net = getNet3()
  numberEpochs = 30
elif source == '3':
  #net = getNet4()
  #net = getNet5()
  #net = getNet6()
  numberEpochs = 5

optimizer = SGD(lr=0.001, momentum=0.95, decay=0.00005, nesterov=True)
#optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
#optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-10)

net.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

predictY = np.array([])
y = np.array([])
for i in range(numberEpochs):
  #This is one epoch
  startIndex = 0
  print('Epoch ' + str(i+1) + ' starting')
  while startIndex < trainingLabels.shape[0]:
    xTrainBatch, yTrainBatch = getNextTrainBatch(startIndex)
    y = np.append(y, yTrainBatch)
    net.train_on_batch(xTrainBatch, yTrainBatch)
    batchLoss = net.test_on_batch(xTrainBatch, yTrainBatch)
    predictYBatch = net.predict_on_batch(xTrainBatch)
    predictY = np.append(predictY, predictYBatch)
    startIndex += batchSize
    print('Batch loss ' + str(batchLoss))
  print('Epoch ' + str(i+1) + ' done')
  #epoch ends

print(metrics.log_loss(y, predictY))

#predict on test data batch by batch
predictY = np.array([])
startIndex = 0
while startIndex < trainingLabels.shape[0]:
  xTrainBatch, yTrainBatch = getNextTrainBatch(startIndex)
  predictYBatch = net.predict_on_batch(xTrainBatch)
  predictY = np.append(predictY, predictYBatch)
  startIndex += batchSize

predictions = pd.DataFrame(predictY, index=files)
predictions.index.name = 'img'
predictions.columns = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
predictions.to_csv(predictionsFile)
