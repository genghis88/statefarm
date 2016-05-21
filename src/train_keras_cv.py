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
imageShape = (1, 40, 30)
batchSize = 256
gcEnabled = False
if source == '1':
  trainingDir = 'data/train_normalized'
  testDir = 'data/test_normalized'
  imageSize = 1200
  imageShape = (1, 40, 30)
  batchSize = 256
elif source == '2':
  trainingDir = 'data/train_normalized_big'
  testDir = 'data/test_normalized_big'
  imageSize = 19200
  imageShape = (1, 160, 120)
  batchSize = 64
elif source == '3':
  trainingDir = 'data/train_normalized_square_big'
  testDir = 'data/test_normalized_square_big'
  imageSize = 14400
  imageShape = (1, 120, 120)
  batchSize = 8
  gcEnabled = True
elif source == '4':
  trainingDir = 'data/train_normalized_80_60'
  testDir = 'data/test_normalized_80_60'
  imageSize = 4800
  imageShape = (1, 80, 60)
  batchSize = 128
elif source == '5':
  trainingDir = 'data/train_normalized_224_224'
  testDir = 'data/test_normalized_224_224'
  imageSize = 50176
  imageShape = (1, 224, 224)
  batchSize = 2
elif source == '6':
  trainingDir = 'data/train_normalized_3_224_224'
  testDir = 'data/test_normalized_3_224_224'
  imageSize = 150528
  imageShape = (3, 224, 224)
  batchSize = 1

trainingLabels = pd.read_csv(trainFile)
print(trainingLabels.shape)

xTrain = np.zeros((trainingLabels.shape[0], imageSize))
for index, row in trainingLabels.iterrows():
  fileName = path.join(trainingDir, row['classname'] + row['img'])
  im = Image.open(fileName)
  xTrain[index, :] = np.reshape(im, imageSize)
  im.close()

xTrain /= 255
xTrain = xTrain.reshape(xTrain.shape[0], imageShape[0], imageShape[1], imageShape[2]).astype('float32')
print(xTrain.shape)

#xTrain /= xTrain.std(axis = None)
#xTrain -= xTrain.mean()
#scaler = StandardScaler()
#scaler.fit(xTrain)
#xTrain = scaler.transform(xTrain)
#xTrain = normalize(xTrain, axis=0)

originalY = np.array([int(x[-1:]) for x in trainingLabels['classname']]).astype('int32')
y = to_categorical(originalY, 10)
print(y.shape)

files = [ f for f in listdir(testDir) if path.isfile(path.join(testDir,f)) ]
xTest = np.zeros((len(files), imageSize), dtype='float32')
index = 0
for fName in files:
  fileName = path.join(testDir,fName)
  im = Image.open(fileName)
  xTest[index, :] = np.reshape(im, imageSize)
  im.close()
  index += 1

xTest /= 255
xTest = xTest.reshape(xTest.shape[0], imageShape[0], imageShape[1], imageShape[2]).astype('float32')

def getNet1():
  net = Sequential()
  net.add(Convolution2D(32, 3, 3, input_shape=(xTrain.shape[1], xTrain.shape[2], xTrain.shape[3]), init='glorot_uniform'))
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
  net.add(ZeroPadding2D((1,1), input_shape=(xTrain.shape[1], xTrain.shape[2], xTrain.shape[3])))
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

def getNet4():
  #learningRate = 0.0001
  #momentum = 0.9
  #decay = 0.00001
  #numberEpochs = 10
  net = Sequential()
  net.add(Convolution2D(32, 7, 7, input_shape=(xTrain.shape[1], xTrain.shape[2], xTrain.shape[3]), init='glorot_uniform')) # 114x114
  net.add(ELU())
  net.add(MaxPooling2D(pool_size=(2,2)))
  net.add(Convolution2D(64, 6, 6, init='glorot_uniform')) # 52x52
  net.add(Activation('tanh'))
  net.add(MaxPooling2D(pool_size=(2,2)))
  net.add(Convolution2D(128, 5, 5, init='glorot_uniform')) # 22x22
  net.add(Activation('tanh'))
  net.add(MaxPooling2D(pool_size=(2,2)))
  net.add(Convolution2D(256, 4, 4, init='glorot_uniform')) # 8x8
  net.add(Activation('tanh'))
  net.add(MaxPooling2D(pool_size=(2,2)))
  #net.add(Convolution2D(512, 3, 3, init='glorot_uniform')) # 2x2
  #net.add(Activation('tanh'))
  net.add(Flatten())
  net.add(Dense(2048, init='glorot_uniform'))
  net.add(ELU())
  net.add(Dropout(0.5))
  net.add(Dense(1024, init='glorot_uniform'))
  net.add(Activation('tanh'))
  net.add(Dropout(0.5))
  net.add(Dense(512, init='glorot_uniform'))
  net.add(Activation('tanh'))
  #net.add(Dropout(0.5))
  #net.add(Dense(256, init='glorot_uniform'))
  #net.add(Activation('tanh'))
  net.add(Dense(10, activation='softmax'))
  return net

def getNet5():
  net = Sequential()
  net.add(Convolution2D(32, 7, 7, input_shape=(xTrain.shape[1], xTrain.shape[2], xTrain.shape[3]), init='glorot_uniform')) # 114x114
  net.add(Activation('relu'))
  net.add(MaxPooling2D(pool_size=(2,2)))
  net.add(Convolution2D(64, 6, 6, init='glorot_uniform')) # 52x52
  net.add(Activation('relu'))
  net.add(MaxPooling2D(pool_size=(2,2)))
  net.add(Convolution2D(128, 5, 5, init='glorot_uniform')) # 22x22
  net.add(Activation('relu'))
  net.add(MaxPooling2D(pool_size=(2,2)))
  net.add(Convolution2D(256, 4, 4, init='glorot_uniform')) # 8x8
  net.add(Activation('relu'))
  net.add(MaxPooling2D(pool_size=(2,2)))
  #net.add(Convolution2D(512, 3, 3, init='glorot_uniform')) # 2x2
  #net.add(Activation('tanh'))
  net.add(Flatten())
  net.add(Dense(2048, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(Dropout(0.5))
  net.add(Dense(1024, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(Dropout(0.5))
  net.add(Dense(512, init='glorot_uniform'))
  net.add(Activation('relu'))
  #net.add(Dropout(0.5))
  #net.add(Dense(256, init='glorot_uniform'))
  #net.add(Activation('relu'))
  net.add(Dense(10, activation='softmax'))
  return net

def getNet6():
  net = Sequential()
  net.add(ZeroPadding2D((1,1), input_shape=(xTrain.shape[1], xTrain.shape[2], xTrain.shape[3]))) #122x122
  net.add(Convolution2D(32, 3, 3, init='glorot_uniform')) #120x120
  net.add(Activation('relu'))
  net.add(ZeroPadding2D((1,1))) #122x122
  net.add(Convolution2D(32, 3, 3, init='glorot_uniform')) #120x120
  net.add(Activation('relu'))
  net.add(Convolution2D(32, 7, 7, init='glorot_uniform')) #114x114
  net.add(Activation('relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #57x57

  net.add(ZeroPadding2D((1,1))) #59x59
  net.add(Convolution2D(64, 3, 3, init='glorot_uniform')) #57x57
  net.add(Activation('relu'))
  net.add(ZeroPadding2D((1,1))) #59x59
  net.add(Convolution2D(64, 3, 3, init='glorot_uniform')) #57x57
  net.add(Activation('relu'))
  net.add(Convolution2D(64, 6, 6, init='glorot_uniform')) #52x52
  net.add(Activation('relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #26x26

  net.add(ZeroPadding2D((1,1))) #28x28
  net.add(Convolution2D(128, 3, 3, init='glorot_uniform')) #26x26
  net.add(Activation('relu'))
  net.add(ZeroPadding2D((1,1))) #28x28
  net.add(Convolution2D(128, 3, 3, init='glorot_uniform')) #26x26
  net.add(Activation('relu'))
  net.add(Convolution2D(128, 5, 5, init='glorot_uniform')) #22x22
  net.add(Activation('relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #11x11

  net.add(ZeroPadding2D((1,1))) #13x13
  net.add(Convolution2D(256, 3, 3, init='glorot_uniform')) #11x11
  net.add(Activation('relu'))
  net.add(ZeroPadding2D((1,1))) #13x13
  net.add(Convolution2D(256, 3, 3, init='glorot_uniform')) #11x11
  net.add(Activation('relu'))
  net.add(Convolution2D(256, 4, 4, init='glorot_uniform')) #8x8
  net.add(Activation('relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #4x4

  #net.add(Convolution2D(512, 3, 3, init='glorot_uniform')) #2x2
  #net.add(Activation('relu'))

  net.add(Flatten())

  net.add(Dense(2048, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(Dropout(0.5))
  net.add(Dense(1024, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(Dropout(0.5))
  net.add(Dense(512, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(Dense(10, activation='softmax'))
  return net

def getNet7():
  net = Sequential()
  net.add(ZeroPadding2D((1,1), input_shape=(imageShape[0], imageShape[1], imageShape[2]))) #122x122
  net.add(Convolution2D(32, 3, 3, init='glorot_uniform')) #120x120
  net.add(Activation('relu'))
  net.add(ZeroPadding2D((1,1))) #122x122
  net.add(Convolution2D(32, 3, 3, init='glorot_uniform')) #120x120
  net.add(Activation('relu'))
  net.add(Convolution2D(32, 7, 7, init='glorot_uniform')) #114x114
  net.add(Activation('relu'))
  net.add(MaxPooling2D(pool_size=(3,3))) #38x38

  net.add(ZeroPadding2D((1,1))) #40x40
  net.add(Convolution2D(64, 3, 3, init='glorot_uniform')) #38x38
  net.add(Activation('relu'))
  net.add(ZeroPadding2D((1,1))) #40x40
  net.add(Convolution2D(64, 3, 3, init='glorot_uniform')) #38x38
  net.add(Activation('relu'))
  net.add(Convolution2D(64, 5, 5, init='glorot_uniform')) #34x34
  net.add(Activation('relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #17x17

  net.add(ZeroPadding2D((1,1))) #19x19
  net.add(Convolution2D(128, 3, 3, init='glorot_uniform')) #17x17
  net.add(Activation('relu'))
  net.add(ZeroPadding2D((1,1))) #19x19
  net.add(Convolution2D(128, 3, 3, init='glorot_uniform')) #17x17
  net.add(Activation('relu'))
  net.add(Convolution2D(128, 4, 4, init='glorot_uniform')) #14x14
  net.add(Activation('relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #7x7

  net.add(ZeroPadding2D((1,1))) #9x9
  net.add(Convolution2D(256, 3, 3, init='glorot_uniform')) #7x7
  net.add(Activation('relu'))
  net.add(ZeroPadding2D((1,1))) #9x9
  net.add(Convolution2D(256, 3, 3, init='glorot_uniform')) #7x7
  net.add(Activation('relu'))
  net.add(Convolution2D(256, 4, 4, init='glorot_uniform')) #4z4
  net.add(Activation('relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #2x2

  net.add(Flatten())

  net.add(Dense(2048, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(Dropout(0.5))
  net.add(Dense(1024, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(Dropout(0.5))
  net.add(Dense(512, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(Dense(10, activation='softmax'))
  return net

def getNet8():
  net = Sequential()
  net.add(ZeroPadding2D((1,1), input_shape=(xTrain.shape[1], xTrain.shape[2], xTrain.shape[3]))) #122x122
  net.add(Convolution2D(32, 3, 3, init='glorot_uniform')) #120x120
  net.add(Activation('relu'))
  net.add(ZeroPadding2D((1,1))) #122x122
  net.add(Convolution2D(32, 3, 3, init='glorot_uniform')) #120x120
  net.add(Activation('relu'))
  net.add(Convolution2D(32, 7, 7, init='glorot_uniform')) #114x114
  net.add(Activation('relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #57x57

  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(64, 3, 3, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(64, 3, 3, init='glorot_uniform')) #57x57
  net.add(Activation('relu'))
  net.add(Convolution2D(64, 6, 6, init='glorot_uniform')) #52x52
  net.add(Activation('relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #26x26

  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(128, 3, 3, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(128, 3, 3, init='glorot_uniform')) #26x26
  net.add(Activation('relu'))
  net.add(Convolution2D(128, 3, 3, init='glorot_uniform')) #24x24
  net.add(Activation('relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #12x12

  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(256, 3, 3, init='glorot_uniform'))
  #net.add(Convolution2D(384, 3, 3, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(256, 3, 3, init='glorot_uniform')) #12x12
  #net.add(Convolution2D(384, 3, 3, init='glorot_uniform')) #6x6
  net.add(Activation('relu'))
  net.add(Convolution2D(256, 4, 4, init='glorot_uniform')) #9x9
  #net.add(Convolution2D(384, 4, 4, init='glorot_uniform')) #4x4
  net.add(Activation('relu'))
  net.add(MaxPooling2D(pool_size=(3,3))) #3x3

  net.add(Flatten())

  '''net.add(Dense(2048, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(Dropout(0.5))
  net.add(Dense(1024, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(Dropout(0.5))
  net.add(Dense(512, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(Dense(10, activation='softmax'))'''

  net.add(Dense(768, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(Dropout(0.5))
  net.add(Dense(384, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(Dropout(0.5))
  net.add(Dense(192, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(Dense(10, activation='softmax'))

  return net

def getNet9():
  net = Sequential()
  net.add(ZeroPadding2D((1,1), input_shape=(xTrain.shape[1], xTrain.shape[2], xTrain.shape[3])))
  net.add(Convolution2D(32, 3, 3, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(32, 3, 3, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(Convolution2D(32, 5, 3, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #38x29

  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(64, 3, 3, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(64, 3, 3, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(Convolution2D(64, 5, 4, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #17x13

  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(128, 3, 3, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(128, 3, 3, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(Convolution2D(128, 4, 4, init='glorot_uniform'))
  net.add(Activation('relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #7x5

  net.add(Convolution2D(256, 4, 4, init='glorot_uniform'))
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

def getNet11():
  net = Sequential()

  net.add(ZeroPadding2D((1,1), input_shape=(imageShape[0], imageShape[1], imageShape[2])))
  net.add(Convolution2D(64, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(64, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #112x112

  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(128, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(128, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #56x56

  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(256, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(256, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(256, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #28x28

  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(512, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(512, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(512, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #14x14

  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(512, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(512, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(512, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #7x7

  net.add(Flatten())

  net.add(Dense(4096, init='glorot_uniform', activation='relu'))
  net.add(Dropout(0.5))
  net.add(Dense(4096, init='glorot_uniform', activation='relu'))
  net.add(Dropout(0.5))
  net.add(Dense(1000, activation='softmax'))

  #net.load_weights('data/vgg16_weights.h5')

  #net.layers.pop()
  net.add(Dense(10, activation='softmax'))

  return net

def getNet12():
  net = Sequential()

  net.add(ZeroPadding2D((1,1), input_shape=(xTrain.shape[1], xTrain.shape[2], xTrain.shape[3])))
  net.add(Convolution2D(64, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(64, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #112x112

  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(128, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(128, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #56x56

  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(256, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(256, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(256, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #28x28

  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(512, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(512, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(512, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #14x14

  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(512, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(512, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(ZeroPadding2D((1,1)))
  net.add(Convolution2D(512, 3, 3, init='glorot_uniform', activation='relu'))
  net.add(MaxPooling2D(pool_size=(2,2))) #7x7

  net.add(Flatten())

  net.add(Dense(4096, init='glorot_uniform', activation='relu'))
  net.add(Dropout(0.5))
  net.add(Dense(4096, init='glorot_uniform', activation='relu'))
  net.add(Dropout(0.5))
  net.add(Dense(1000, activation='softmax'))

  net.load_weights('data/vgg16_weights.h5')

  net.layers.pop()
  net.add(Dense(10, activation='softmax'))

  return net

def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
  data = []
  target = []
  index = []
  for i in range(len(driver_id)):
      if driver_id[i] in driver_list:
          data.append(train_data[i])
          target.append(train_target[i])
          index.append(i)
  data = np.array(data, dtype=np.float32)
  target = np.array(target, dtype=np.float32)
  index = np.array(index, dtype=np.uint32)
  return data, target, index

def merge_several_folds_mean(data, nfolds):
  a = np.array(data[0])
  b = np.array(data[0])
  for i in range(1, nfolds):
    a += np.array(data[i])
    b *= np.array(data[i])
  a /= nfolds
  b = np.power(b, (1. / nfolds))
  return a.tolist(), b.tolist()

def dict_to_list(d):
  ret = []
  for i in d.items():
      ret.append(i[1])
  return ret

def run_cross_validation(nfolds=10):
  unique_drivers = trainingLabels['subject'].unique()
  #driver_id = {}
  #for d_id in unique_drivers:
    #driver_id[d_id] = trainingLabels.loc[trainingLabels['subject'] == d_id]['img']
  driver_id = trainingLabels['subject']
  #print('Unique drivers ' + str(unique_drivers) + '\n' + str(driver_id))
  kf = cross_validation.KFold(len(unique_drivers), n_folds=nfolds, shuffle=True)
  num_fold = 1
  yfull_train = dict()
  yfull_test = []
  for train_drivers, test_drivers in kf:
    unique_list_train = [unique_drivers[i] for i in train_drivers]
    #print('Unique drivers train ' + str(unique_list_train))
    X_train, Y_train, train_index = copy_selected_drivers(xTrain, y, driver_id, unique_list_train)
    unique_list_valid = [unique_drivers[i] for i in test_drivers]
    #print('Unique drivers validation ' + str(unique_list_valid))
    X_valid, Y_valid, test_index = copy_selected_drivers(xTrain, y, driver_id, unique_list_valid)

    net = None
    if source == '1':
      net = getNet1()
      #net = getNet2()
      numberEpochs = 30
    elif source == '2':
      net = getNet3()
      numberEpochs = 30
    elif source == '3':
      #net = getNet4()
      #net = getNet5()
      #net = getNet6()
      #net = getNet7()
      net = getNet8()
      numberEpochs = 5
    elif source == '4':
      net = getNet9()
      #net = getNet10()
      numberEpochs = 15
    elif source == '5':
      net = getNet11()
      numberEpochs = 5
    elif source == '6':
      net = getNet12()
      numberEpochs = 5

    #optimizer = SGD(lr=0.0001, momentum=0.95, decay=0.00001, nesterov=True)
    optimizer = SGD(lr=0.001, momentum=0.95, decay=0.00005, nesterov=True)
    #optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
    #optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-10)

    net.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    net.fit(X_train, Y_train,
        nb_epoch=numberEpochs,
        batch_size=batchSize,
        verbose=1,
        validation_data=(X_valid, Y_valid))

    predictValidY = net.predict_proba(X_valid)
    score = metrics.log_loss(Y_valid, predictValidY)
    print('Fold ' + str(num_fold) + ' score ' + str(score))

    for i in range(len(test_index)):
      yfull_train[test_index[i]] = predictValidY[i]

    test_prediction = net.predict_proba(xTest)
    yfull_test.append(test_prediction)

    if gcEnabled:
      gc.collect()

    num_fold += 1

  score = metrics.log_loss(y, dict_to_list(yfull_train))
  print('Final log loss ' + str(score))
  test_res_am, test_res_gm = merge_several_folds_mean(yfull_test, nfolds)
  predictions = pd.DataFrame(test_res_am, index=files)
  predictions.index.name = 'img'
  predictions.columns = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
  predictions.to_csv(predictionsFile + '_am')
  predictions = pd.DataFrame(test_res_gm, index=files)
  predictions.index.name = 'img'
  predictions.columns = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
  predictions.to_csv(predictionsFile + '_gm')

run_cross_validation(10)

'''if source == '1':
  #net = getNet1()
  net = getNet2()
  numberEpochs = 10
elif source == '2':
  net = getNet3()
  numberEpochs = 30
elif source == '3':
  #net = getNet4()
  #net = getNet5()
  net = getNet6()
  numberEpochs = 5

optimizer = SGD(lr=0.01, momentum=0.9, decay=0.0002, nesterov=True)
#optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
#optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-10)

net.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

x_fit, x_eval, y_fit, y_eval= cross_validation.train_test_split(xTrain, y, test_size=0.2)

net.fit(x_fit, y_fit,
        nb_epoch=numberEpochs,
        batch_size=batchSize,
        verbose=1,
        validation_data=(x_eval, y_eval))

predictY = net.predict_proba(xTrain)
print(metrics.log_loss(originalY, predictY))

files = [ f for f in listdir(testDir) if path.isfile(path.join(testDir,f)) ]
xTest = np.zeros((len(files), imageSize), dtype='float32')
index = 0
for fName in files:
  fileName = path.join(testDir,fName)
  im = Image.open(fileName)
  xTest[index, :] = np.reshape(im, imageSize)
  im.close()
  index += 1

xTest /= 255
xTest = xTest.reshape(xTest.shape[0], 1, imageShape[0], imageShape[1]).astype('float32')

predictY = net.predict_proba(xTest)

predictions = pd.DataFrame(predictY, index=files)
predictions.index.name = 'img'
predictions.columns = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
predictions.to_csv(predictionsFile)'''

#with open(pickleFile,'wb') as f:
#  sys.setrecursionlimit(20000)
#  pickle.dump(net, f)