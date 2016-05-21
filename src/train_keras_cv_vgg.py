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

gcEnabled = False
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

originalY = np.array([int(x[-1:]) for x in trainingLabels['classname']]).astype('int32')
y = to_categorical(originalY, 10)
print(y.shape)

files = [ f for f in listdir(testDir) if path.isfile(path.join(testDir,f)) ]

def getNet12():
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
  xTestBatch = []
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

def run_cross_validation(nfolds=10):
  unique_drivers = trainingLabels['subject'].unique()
  driver_id = trainingLabels['subject']
  kf = cross_validation.KFold(len(unique_drivers), n_folds=nfolds, shuffle=True)
  num_fold = 1
  yfull_train = dict()
  yfull_test = []
  for train_drivers, test_drivers in kf:
    unique_list_train = [unique_drivers[i] for i in train_drivers]
    print('Unique drivers train ' + str(unique_list_train))
    #X_train, Y_train, train_index = copy_selected_drivers(xTrain, y, driver_id, unique_list_train)
    X_train, Y_train, train_index = getNextTrainBatch(xTrain, y, driver_id, unique_list_train)
    unique_list_valid = [unique_drivers[i] for i in test_drivers]
    print('Unique drivers validation ' + str(unique_list_valid))
    X_valid, Y_valid, test_index = copy_selected_drivers(xTrain, y, driver_id, unique_list_valid)

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