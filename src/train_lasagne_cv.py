import pandas as pd
from os import path
from os import listdir
from PIL import Image
import pickle
import sys
import numpy as np
from lasagne import layers
from lasagne.init import Constant
from lasagne import objectives
from lasagne import updates
from lasagne.nonlinearities import softmax, sigmoid, rectify, tanh, ScaledTanH, elu, identity, softplus, leaky_rectify
from lasagne.init import GlorotUniform, HeUniform, GlorotNormal, HeNormal
from nolearn.lasagne import NeuralNet, BatchIterator, TrainSplit
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
gcEnabled = False
if source == '1':
  trainingDir = 'data/train_normalized'
  testDir = 'data/test_normalized'
  imageSize = 1200
  imageShape = (40, 30)
  batchSize = 256
elif source == '2':
  trainingDir = 'data/train_normalized_big'
  testDir = 'data/test_normalized_big'
  imageSize = 19200
  imageShape = (160, 120)
  batchSize = 64
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

xTrain = np.zeros((trainingLabels.shape[0], imageSize))
for index, row in trainingLabels.iterrows():
  fileName = path.join(trainingDir, row['classname'] + row['img'])
  im = Image.open(fileName)
  xTrain[index, :] = np.reshape(im, imageSize)
  im.close()

xTrain /= 255
xTrain = xTrain.reshape(xTrain.shape[0], 1, imageShape[0], imageShape[1]).astype('float32')
print(xTrain.shape)

#xTrain /= xTrain.std(axis = None)
#xTrain -= xTrain.mean()
#scaler = StandardScaler()
#scaler.fit(xTrain)
#xTrain = scaler.transform(xTrain)
#xTrain = normalize(xTrain, axis=0)

originalY = np.array([int(x[-1:]) for x in trainingLabels['classname']]).astype('int32')
y = originalY
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
xTest = xTest.reshape(xTest.shape[0], 1, imageShape[0], imageShape[1]).astype('float32')

def getNet1():
  inputLayer = layers.InputLayer(shape=(None, 1, imageShape[0], imageShape[1]))
  conv1Layer = layers.Conv2DLayer(inputLayer, num_filters=32, filter_size=(3,3), W=GlorotNormal('relu'), nonlinearity=rectify)
  pool1Layer = layers.MaxPool2DLayer(conv1Layer, pool_size=(2,2))
  conv2Layer = layers.Conv2DLayer(pool1Layer, num_filters=64, filter_size=(4,3), W=GlorotNormal('relu'), nonlinearity=rectify)
  pool2Layer = layers.MaxPool2DLayer(conv2Layer, pool_size=(2,2))
  #conv3Layer = layers.Conv2DLayer(pool2Layer, num_filters=128, filter_size=(3,3), W=GlorotNormal('relu'), nonlinearity=rectify)
  conv3Layer = layers.Conv2DLayer(pool2Layer, num_filters=128, filter_size=(5,4), W=GlorotNormal('relu'), nonlinearity=rectify)
  #pool3Layer = layers.MaxPool2DLayer(conv3Layer, pool_size=(2,2))
  #conv4Layer = layers.Conv2DLayer(pool3Layer, num_filters=256, filter_size=(3,2), W=GlorotNormal('relu'), nonlinearity=rectify)
  hidden1Layer = layers.DenseLayer(conv3Layer, num_units=1024, W=GlorotNormal('relu'), nonlinearity=rectify)
  dropout1Layer = layers.DropoutLayer(hidden1Layer, p=0.5)
  hidden2Layer = layers.DenseLayer(dropout1Layer, num_units=512, W=GlorotNormal('relu'), nonlinearity=rectify)
  dropout2Layer = layers.DropoutLayer(hidden2Layer, p=0.5)
  hidden3Layer = layers.DenseLayer(dropout2Layer, num_units=256, W=GlorotNormal('relu'), nonlinearity=rectify)
  outputLayer = layers.DenseLayer(hidden3Layer, num_units=10, W=GlorotNormal(1.0), nonlinearity=softmax)
  return outputLayer

def getNet2():
  inputLayer = layers.InputLayer(shape=(None, 1, imageShape[0], imageShape[1]))
  loc1Layer = layers.Conv2DLayer(inputLayer, num_filters=32, filter_size=(3,3), W=GlorotUniform('relu'), nonlinearity=rectify)
  loc2Layer = layers.MaxPool2DLayer(loc1Layer, pool_size=(2,2))
  loc3Layer = layers.Conv2DLayer(loc2Layer, num_filters=64, filter_size=(4,3), W=GlorotUniform('relu'), nonlinearity=rectify)
  loc4Layer = layers.MaxPool2DLayer(loc3Layer, pool_size=(2,2))
  loc5Layer = layers.Conv2DLayer(loc4Layer, num_filters=128, filter_size=(3,3), W=GlorotUniform('relu'), nonlinearity=rectify)
  loc6Layer = layers.MaxPool2DLayer(loc5Layer, pool_size=(2,2))
  loc7Layer = layers.Conv2DLayer(loc6Layer, num_filters=256, filter_size=(3,2), W=GlorotUniform('relu'), nonlinearity=rectify)
  #loc7Layer = layers.DenseLayer(loc5Layer, num_units=1024, nonlinearity=rectify)
  loc8Layer = layers.DenseLayer(loc7Layer, num_units=256, W=GlorotUniform('relu'), nonlinearity=rectify)
  loc9Layer = layers.DenseLayer(loc8Layer, num_units=128, W=GlorotUniform('relu'), nonlinearity=rectify)
  loc10Layer = layers.DenseLayer(loc9Layer, num_units=64, W=GlorotUniform('relu'), nonlinearity=rectify)
  #loc11Layer = layers.DenseLayer(loc10Layer, num_units=32, nonlinearity=tanh)
  #loc12Layer = layers.DenseLayer(loc11Layer, num_units=16, nonlinearity=tanh)
  locOutLayer = layers.DenseLayer(loc10Layer, num_units=6, W=GlorotUniform(1.0), nonlinearity=identity)

  transformLayer = layers.TransformerLayer(inputLayer, locOutLayer, downsample_factor=1.0)

  conv1Layer = layers.Conv2DLayer(transformLayer, num_filters=32, filter_size=(3,3), W=GlorotNormal('relu'), nonlinearity=rectify)
  pool1Layer = layers.MaxPool2DLayer(conv1Layer, pool_size=(2,2))
  conv2Layer = layers.Conv2DLayer(pool1Layer, num_filters=64, filter_size=(4,3), W=GlorotUniform('relu'), nonlinearity=rectify)
  pool2Layer = layers.MaxPool2DLayer(conv2Layer, pool_size=(2,2))
  conv3Layer = layers.Conv2DLayer(pool2Layer, num_filters=128, filter_size=(3,3), W=GlorotUniform('relu'), nonlinearity=rectify)
  pool3Layer = layers.MaxPool2DLayer(conv3Layer, pool_size=(2,2))
  conv4Layer = layers.Conv2DLayer(pool3Layer, num_filters=256, filter_size=(3,2), W=GlorotNormal('relu'), nonlinearity=rectify)
  hidden1Layer = layers.DenseLayer(conv4Layer, num_units=1024, W=GlorotUniform('relu'), nonlinearity=rectify)
  dropout1Layer = layers.DropoutLayer(hidden1Layer, p=0.5)
  hidden2Layer = layers.DenseLayer(dropout1Layer, num_units=512, W=GlorotUniform('relu'), nonlinearity=rectify)
  #hidden3Layer = layers.DenseLayer(hidden2Layer, num_units=256, nonlinearity=tanh)
  outputLayer = layers.DenseLayer(hidden2Layer, num_units=10, W=GlorotUniform(1.0), nonlinearity=softmax)
  return outputLayer

def getNet3():
  inputLayer = layers.InputLayer(shape=(None, 1, imageShape[0], imageShape[1]))
  conv1Layer = layers.Conv2DLayer(inputLayer, num_filters=32, filter_size=(3,3), nonlinearity=elu)
  pool1Layer = layers.MaxPool2DLayer(conv1Layer, pool_size=(2,2))
  dropout1Layer = layers.DropoutLayer(pool1Layer, p=0.2)
  conv2Layer = layers.Conv2DLayer(dropout1Layer, num_filters=64, filter_size=(4,3), nonlinearity=tanh)
  pool2Layer = layers.MaxPool2DLayer(conv2Layer, pool_size=(2,2))
  dropout2Layer = layers.DropoutLayer(pool2Layer, p=0.2)
  conv3Layer = layers.Conv2DLayer(dropout2Layer, num_filters=128, filter_size=(3,3), nonlinearity=tanh)
  pool3Layer = layers.MaxPool2DLayer(conv3Layer, pool_size=(2,2))
  dropout3Layer = layers.DropoutLayer(pool3Layer, p=0.2)
  conv4Layer = layers.Conv2DLayer(dropout3Layer, num_filters=256, filter_size=(3,2), nonlinearity=elu)
  hidden1Layer = layers.DenseLayer(conv4Layer, num_units=1024, nonlinearity=elu)
  hidden2Layer = layers.DenseLayer(hidden1Layer, num_units=512, nonlinearity=tanh)
  hidden3Layer = layers.DenseLayer(hidden2Layer, num_units=256, nonlinearity=tanh)
  #hidden4Layer = layers.DenseLayer(hidden3Layer, num_units=256, nonlinearity=elu)
  #hidden5Layer = layers.DenseLayer(hidden4Layer, num_units=128, nonlinearity=tanh)
  outputLayer = layers.DenseLayer(hidden3Layer, num_units=10, nonlinearity=softmax)
  return outputLayer

def getNet4():
  inputLayer = layers.InputLayer(shape=(None, 1, imageShape[0], imageShape[1])) #120x120
  conv1Layer = layers.Conv2DLayer(inputLayer, num_filters=32, filter_size=(5,5), nonlinearity=elu) #116x116
  pool1Layer = layers.MaxPool2DLayer(conv1Layer, pool_size=(2,2)) #58x58
  conv2Layer = layers.Conv2DLayer(pool1Layer, num_filters=64, filter_size=(5,5), nonlinearity=tanh) #54x54
  pool2Layer = layers.MaxPool2DLayer(conv2Layer, pool_size=(2,2)) #27x27
  conv3Layer = layers.Conv2DLayer(pool2Layer, num_filters=128, filter_size=(4,4), nonlinearity=tanh) #24x24
  pool3Layer = layers.MaxPool2DLayer(conv3Layer, pool_size=(2,2)) #12x12
  conv4Layer = layers.Conv2DLayer(pool3Layer, num_filters=256, filter_size=(3,3), nonlinearity=elu) #10x10
  pool4Layer = layers.MaxPool2DLayer(conv4Layer, pool_size=(2,2)) #5x5
  conv5Layer = layers.Conv2DLayer(pool4Layer, num_filters=512, filter_size=(4,4), nonlinearity=tanh) #2x2
  hidden1Layer = layers.DenseLayer(conv5Layer, num_units=2048, nonlinearity=tanh)
  dropout1Layer = layers.DropoutLayer(hidden1Layer, p=0.5)
  hidden2Layer = layers.DenseLayer(dropout1Layer, num_units=1024, nonlinearity=elu)
  dropout2Layer = layers.DropoutLayer(hidden2Layer, p=0.5)
  hidden3Layer = layers.DenseLayer(dropout2Layer, num_units=512, nonlinearity=tanh)
  dropout3Layer = layers.DropoutLayer(hidden3Layer, p=0.5)
  hidden4Layer = layers.DenseLayer(dropout3Layer, num_units=256, nonlinearity=tanh)
  outputLayer = layers.DenseLayer(hidden4Layer, num_units=10, nonlinearity=softmax)
  return outputLayer

def getNet5():
  inputLayer = layers.InputLayer(shape=(None, 1, imageShape[0], imageShape[1])) #120x120
  conv1Layer = layers.Conv2DLayer(inputLayer, num_filters=32, filter_size=(4,4), nonlinearity=elu) #117x117
  pool1Layer = layers.MaxPool2DLayer(conv1Layer, pool_size=(3,3)) #39x39
  conv2Layer = layers.Conv2DLayer(pool1Layer, num_filters=64, filter_size=(4,4), nonlinearity=tanh) #36x36
  pool2Layer = layers.MaxPool2DLayer(conv2Layer, pool_size=(2,2)) #18x18
  conv3Layer = layers.Conv2DLayer(pool2Layer, num_filters=128, filter_size=(4,4), nonlinearity=sigmoid) #15x15
  pool3Layer = layers.MaxPool2DLayer(conv3Layer, pool_size=(3,3)) #5x5
  conv4Layer = layers.Conv2DLayer(pool3Layer, num_filters=256, filter_size=(4,4), nonlinearity=tanh) #2x2
  hidden1Layer = layers.DenseLayer(conv4Layer, num_units=1024, nonlinearity=elu)
  dropout1Layer = layers.DropoutLayer(hidden1Layer, p=0.5)
  hidden2Layer = layers.DenseLayer(dropout1Layer, num_units=512, nonlinearity=tanh)
  dropout2Layer = layers.DropoutLayer(hidden2Layer, p=0.5)
  hidden3Layer = layers.DenseLayer(dropout2Layer, num_units=256, nonlinearity=tanh)
  dropout3Layer = layers.DropoutLayer(hidden3Layer, p=0.5)
  hidden4Layer = layers.DenseLayer(dropout3Layer, num_units=128, nonlinearity=tanh)
  outputLayer = layers.DenseLayer(hidden4Layer, num_units=10, nonlinearity=softmax)
  return outputLayer

def getNet6():
  inputLayer = layers.InputLayer(shape=(None, 1, imageShape[0], imageShape[1])) #120x120

  conv1Layer = layers.Conv2DLayer(inputLayer, num_filters=32, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify) #120x120
  conv2Layer = layers.Conv2DLayer(conv1Layer, num_filters=32, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify) #120x120
  pool1Layer = layers.MaxPool2DLayer(conv2Layer, pool_size=(2,2)) #60x60
  conv3Layer = layers.Conv2DLayer(pool1Layer, num_filters=64, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify) #60x60
  conv4Layer = layers.Conv2DLayer(conv3Layer, num_filters=64, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify) #60x60
  conv5Layer = layers.Conv2DLayer(conv4Layer, num_filters=64, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify) #60x60
  pool2Layer = layers.MaxPool2DLayer(conv5Layer, pool_size=(2,2)) #30x30
  conv6Layer = layers.Conv2DLayer(pool2Layer, num_filters=128, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify) #30x30
  conv7Layer = layers.Conv2DLayer(conv6Layer, num_filters=128, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify) #30x30
  conv8Layer = layers.Conv2DLayer(conv7Layer, num_filters=128, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify) #30x30
  pool3Layer = layers.MaxPool2DLayer(conv8Layer, pool_size=(2,2)) #15x15
  conv9Layer = layers.Conv2DLayer(pool3Layer, num_filters=256, filter_size=(4,4), W=HeNormal('relu'), nonlinearity=rectify) #12x12
  flattenLayer = layers.FlattenLayer(conv9Layer)
  hidden1Layer = layers.DenseLayer(flattenLayer, num_units=1024, W=HeNormal('relu'), nonlinearity=rectify)
  dropout1Layer = layers.DropoutLayer(hidden1Layer, p=0.5)
  hidden2Layer = layers.DenseLayer(dropout1Layer, num_units=512, W=HeNormal('relu'), nonlinearity=rectify)
  dropout2Layer = layers.DropoutLayer(hidden2Layer, p=0.5)
  hidden3Layer = layers.DenseLayer(dropout2Layer, num_units=256, W=HeNormal('relu'), nonlinearity=rectify)
  dropout3Layer = layers.DropoutLayer(hidden3Layer, p=0.5)
  hidden4Layer = layers.DenseLayer(dropout3Layer, num_units=128, W=HeNormal('relu'), nonlinearity=rectify)
  outputLayer = layers.DenseLayer(hidden4Layer, num_units=10, W=HeNormal('relu'), nonlinearity=softmax)
  return outputLayer

def getNet7():
  inputLayer = layers.InputLayer(shape=(None, 1, imageShape[0], imageShape[1])) #120x120

  conv1Layer = layers.Conv2DLayer(inputLayer, num_filters=32, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify) #120x120
  conv2Layer = layers.Conv2DLayer(conv1Layer, num_filters=32, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify) #120x120
  pool1Layer = layers.MaxPool2DLayer(conv2Layer, pool_size=(2,2)) #60x60
  conv3Layer = layers.Conv2DLayer(pool1Layer, num_filters=64, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify) #60x60
  conv4Layer = layers.Conv2DLayer(conv3Layer, num_filters=64, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify) #60x60
  pool2Layer = layers.MaxPool2DLayer(conv4Layer, pool_size=(2,2)) #30x30
  conv5Layer = layers.Conv2DLayer(pool2Layer, num_filters=128, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify) #30x30
  conv6Layer = layers.Conv2DLayer(conv5Layer, num_filters=128, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify) #30x30
  pool3Layer = layers.MaxPool2DLayer(conv6Layer, pool_size=(2,2)) #15x15
  conv7Layer = layers.Conv2DLayer(pool3Layer, num_filters=256, filter_size=(4,4), W=HeNormal('relu'), nonlinearity=rectify) #12x12
  flattenLayer = layers.FlattenLayer(conv7Layer)
  hidden1Layer = layers.DenseLayer(flattenLayer, num_units=1024, W=HeNormal('relu'), nonlinearity=rectify)
  dropout1Layer = layers.DropoutLayer(hidden1Layer, p=0.5)
  hidden2Layer = layers.DenseLayer(dropout1Layer, num_units=512, W=HeNormal('relu'), nonlinearity=rectify)
  dropout2Layer = layers.DropoutLayer(hidden2Layer, p=0.5)
  hidden3Layer = layers.DenseLayer(dropout2Layer, num_units=256, W=HeNormal('relu'), nonlinearity=rectify)
  #dropout3Layer = layers.DropoutLayer(hidden3Layer, p=0.5)
  #hidden4Layer = layers.DenseLayer(dropout2Layer, num_units=128, W=HeNormal('relu'), nonlinearity=rectify)
  outputLayer = layers.DenseLayer(hidden3Layer, num_units=10, W=HeNormal('relu'), nonlinearity=softmax)
  return outputLayer

def getNet8():
  inputLayer = layers.InputLayer(shape=(None, 1, imageShape[0], imageShape[1]))
  conv1Layer = layers.Conv2DLayer(inputLayer, num_filters=32, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify)
  conv2Layer = layers.Conv2DLayer(conv1Layer, num_filters=32, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify)
  conv3Layer = layers.Conv2DLayer(conv2Layer, num_filters=32, filter_size=(3,3), W=GlorotNormal('relu'), nonlinearity=rectify)
  pool1Layer = layers.MaxPool2DLayer(conv3Layer, pool_size=(2,2))
  conv4Layer = layers.Conv2DLayer(pool1Layer, num_filters=64, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify)
  conv5Layer = layers.Conv2DLayer(conv4Layer, num_filters=64, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify)
  conv6Layer = layers.Conv2DLayer(conv5Layer, num_filters=64, filter_size=(4,3), W=GlorotNormal('relu'), nonlinearity=rectify)
  pool2Layer = layers.MaxPool2DLayer(conv6Layer, pool_size=(2,2))
  conv7Layer = layers.Conv2DLayer(pool2Layer, num_filters=128, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify)
  conv8Layer = layers.Conv2DLayer(conv7Layer, num_filters=128, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify)
  conv9Layer = layers.Conv2DLayer(conv8Layer, num_filters=128, filter_size=(3,3), W=GlorotNormal('relu'), nonlinearity=rectify)
  pool3Layer = layers.MaxPool2DLayer(conv9Layer, pool_size=(2,2))
  #conv4Layer = layers.Conv2DLayer(pool3Layer, num_filters=256, filter_size=(3,2), W=GlorotNormal('relu'), nonlinearity=elu)
  hidden1Layer = layers.DenseLayer(pool3Layer, num_units=1024, W=GlorotNormal('relu'), nonlinearity=rectify)
  dropout1Layer = layers.DropoutLayer(hidden1Layer, p=0.5)
  hidden2Layer = layers.DenseLayer(dropout1Layer, num_units=512, W=GlorotNormal('relu'), nonlinearity=rectify)
  dropout2Layer = layers.DropoutLayer(hidden2Layer, p=0.5)
  hidden3Layer = layers.DenseLayer(dropout2Layer, num_units=256, W=GlorotNormal('relu'), nonlinearity=rectify)
  outputLayer = layers.DenseLayer(hidden3Layer, num_units=10, W=GlorotNormal(1.0), nonlinearity=softmax)
  return outputLayer

def getNet9():
  inputLayer = layers.InputLayer(shape=(None, 1, imageShape[0], imageShape[1]))
  conv1Layer = layers.Conv2DLayer(inputLayer, num_filters=32, filter_size=(5,3), W=GlorotNormal('relu'), nonlinearity=rectify)
  pool1Layer = layers.MaxPool2DLayer(conv1Layer, pool_size=(2,2))
  conv2Layer = layers.Conv2DLayer(pool1Layer, num_filters=64, filter_size=(5,4), W=GlorotNormal('relu'), nonlinearity=rectify)
  pool2Layer = layers.MaxPool2DLayer(conv2Layer, pool_size=(2,2))
  conv3Layer = layers.Conv2DLayer(pool2Layer, num_filters=128, filter_size=(4,4), W=GlorotNormal('relu'), nonlinearity=rectify)
  pool3Layer = layers.MaxPool2DLayer(conv3Layer, pool_size=(2,2))
  conv4Layer = layers.Conv2DLayer(pool3Layer, num_filters=256, filter_size=(4,4), W=GlorotNormal('relu'), nonlinearity=rectify)
  hidden1Layer = layers.DenseLayer(conv4Layer, num_units=2048, W=GlorotNormal('relu'), nonlinearity=rectify)
  dropout1Layer = layers.DropoutLayer(hidden1Layer, p=0.5)
  hidden2Layer = layers.DenseLayer(dropout1Layer, num_units=1024, W=GlorotNormal('relu'), nonlinearity=rectify)
  dropout2Layer = layers.DropoutLayer(hidden2Layer, p=0.5)
  hidden3Layer = layers.DenseLayer(dropout2Layer, num_units=512, W=GlorotNormal('relu'), nonlinearity=rectify)
  outputLayer = layers.DenseLayer(hidden3Layer, num_units=10, W=GlorotNormal(1.0), nonlinearity=softmax)
  return outputLayer

def getNet10():
  inputLayer = layers.InputLayer(shape=(None, 1, imageShape[0], imageShape[1]))
  conv1Layer = layers.Conv2DLayer(inputLayer, num_filters=32, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify)
  conv2Layer = layers.Conv2DLayer(conv1Layer, num_filters=32, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify)
  pool1Layer = layers.MaxPool2DLayer(conv2Layer, pool_size=(2,2))
  conv3Layer = layers.Conv2DLayer(pool1Layer, num_filters=64, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify)
  conv4Layer = layers.Conv2DLayer(conv3Layer, num_filters=64, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify)
  pool2Layer = layers.MaxPool2DLayer(conv4Layer, pool_size=(2,2))
  conv5Layer = layers.Conv2DLayer(pool2Layer, num_filters=128, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify)
  conv6Layer = layers.Conv2DLayer(conv5Layer, num_filters=128, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify)
  conv7Layer = layers.Conv2DLayer(conv6Layer, num_filters=128, filter_size=(5,4), W=GlorotNormal('relu'), nonlinearity=rectify)
  pool3Layer = layers.MaxPool2DLayer(conv7Layer, pool_size=(2,2))
  conv8Layer = layers.Conv2DLayer(pool3Layer, num_filters=256, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify)
  conv9Layer = layers.Conv2DLayer(conv8Layer, num_filters=256, filter_size=(3,3), pad=(1,1), W=HeNormal('relu'), nonlinearity=rectify)
  conv10Layer = layers.Conv2DLayer(conv9Layer, num_filters=256, filter_size=(5,5), W=GlorotNormal('relu'), nonlinearity=rectify)
  hidden1Layer = layers.DenseLayer(conv10Layer, num_units=2048, W=GlorotNormal('relu'), nonlinearity=rectify)
  dropout1Layer = layers.DropoutLayer(hidden1Layer, p=0.5)
  hidden2Layer = layers.DenseLayer(dropout1Layer, num_units=1024, W=GlorotNormal('relu'), nonlinearity=rectify)
  dropout2Layer = layers.DropoutLayer(hidden2Layer, p=0.5)
  hidden3Layer = layers.DenseLayer(dropout2Layer, num_units=512, W=GlorotNormal('relu'), nonlinearity=rectify)
  outputLayer = layers.DenseLayer(hidden3Layer, num_units=10, W=GlorotNormal(1.0), nonlinearity=softmax)
  return outputLayer

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
  for i in range(1, nfolds):
    #a += np.array(data[i])
    a *= np.array(data[i])
  #a /= nfolds
  a = np.power(a, (1. / nfolds))
  return a.tolist()

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

    outputLayer = None
    if source == '1':
      outputLayer = getNet1()
      #outputLayer = getNet2()
      #outputLayer = getNet8()
      numberEpochs = 30
    elif source == '2':
      outputLayer = getNet3()
      numberEpochs = 30
    elif source == '3':
      outputLayer = getNet4()
      #outputLayer = getNet5()
      #outputLayer = getNet6()
      #outputLayer = getNet7()
      numberEpochs = 10
    elif source == '4':
      outputLayer = getNet9()
      #outputLayer = getNet10()
      numberEpochs = 30

    net = NeuralNet(
      layers = outputLayer,
      update=updates.nesterov_momentum,
      #update=updates.adam,
      #update=updates.rmsprop,
      #update=updates.adadelta,
      update_learning_rate = 0.001,
      #update_beta1 = 0.9,
      #update_beta2 = 0.999,
      #update_epsilon = 1e-8,
      update_momentum = 0.9,
      #update_rho=0.95,
      #update_epsilon=1e-06,

      objective_loss_function = objectives.categorical_crossentropy,
      #objective=objectives.categorical_crossentropy,

      batch_iterator_train = BatchIterator(batch_size = batchSize),
      batch_iterator_test = BatchIterator(batch_size = batchSize),
      #custom_scores = [objectives.categorical_accuracy],

      use_label_encoder = True,
      #use_label_encoder = False,
      regression = False,
      max_epochs = numberEpochs,
      verbose = 1
    )

    net.fit(X_train, Y_train)

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
  test_res = merge_several_folds_mean(yfull_test, nfolds)
  predictions = pd.DataFrame(test_res, index=files)
  predictions.index.name = 'img'
  predictions.columns = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
  predictions.to_csv(predictionsFile)

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