import pandas as pd
from os import path
from os import listdir
from PIL import Image
import pickle
import sys
import numpy as np
import xgboost as xgb
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn import cross_validation
import gc
from scipy import signal

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

def getConvolvedImage(allImagesArray):
  newImagesArray = np.zeros(shape=(allImagesArray.shape[0], (imageShape[0] - 8) * (imageShape[1] - 8)))
  for imageIndex in range(len(allImagesArray)):
    image = allImagesArray[imageIndex]
    image = image.reshape(imageShape[0], imageShape[1]).astype('float32')
    filter = np.array([[1./9, 1./9, 1./9], [1./9, 1./9, 1./9], [1./9, 1./9, 1./9]])
    image = signal.convolve2d(image, filter, mode='valid')
    image = signal.convolve2d(image, filter, mode='valid')
    image = signal.convolve2d(image, filter, mode='valid')
    image = signal.convolve2d(image, filter, mode='valid')
    image = image.reshape((imageShape[0] - 8) * (imageShape[1] - 8),).astype('float32')
    newImagesArray[imageIndex] = image
  return allImagesArray


xTrain /= 255
xTrain = getConvolvedImage(xTrain)
#xTrain = xTrain.reshape(xTrain.shape[0], imageShape[0], imageShape[1]).astype('float32')
#xTrainC = theano.shared(xTrain, name='xTrainC')
#w_bound = 9
#W = theano.shared(np.asarray(np.random.uniform(low=-1.0 / w_bound, high=1.0 / w_bound, size=(3,3))), name='W')
#xTrainC = theano.tensor.signal.conv.conv2d(xTrain, W)
#xTrainC = theano.tensor.signal.pool.pool_2d(xTrainC, (2,2), ignore_border=False, mode='max')
#xTrainC = theano.tensor.signal.conv.conv2d(xTrainC, W)
#xTrainC = theano.tensor.signal.pool.pool_2d(xTrainC, (2,2), ignore_border=False, mode='max')
#xTrainC = theano.tensor.signal.conv.conv2d(xTrainC, W)
#xTrainC = theano.tensor.signal.pool.pool_2d(xTrainC, (2,2), ignore_border=False, mode='max')
#xTrain = theano.tensor.reshape(xTrainC, (xTrain.shape[0], 8))
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
#xTest = xTest.reshape(xTest.shape[0], 1, imageShape[0], imageShape[1]).astype('float32')
xTest = getConvolvedImage(xTest)

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
    #X_train, Y_train, train_index = copy_selected_drivers(xTrainC, y, driver_id, unique_list_train)
    unique_list_valid = [unique_drivers[i] for i in test_drivers]
    #print('Unique drivers validation ' + str(unique_list_valid))
    X_valid, Y_valid, test_index = copy_selected_drivers(xTrain, y, driver_id, unique_list_valid)
    #X_valid, Y_valid, test_index = copy_selected_drivers(xTrainC, y, driver_id, unique_list_valid)

    clf = None
    if source == '1':
      clf = xgb.XGBClassifier(objective='multi:softmax', n_estimators=200, learning_rate=0.05, max_depth=5, nthread=4, subsample=0.7, colsample_bytree=0.85, seed=1234)
      numberEpochs = 30
    elif source == '2':
      clf = xgb.XGBClassifier(objective='multi:softmax', n_estimators=200, learning_rate=0.05, max_depth=20, nthread=4, subsample=0.7, colsample_bytree=0.85, seed=2471)
      numberEpochs = 30
    elif source == '3':
      clf = xgb.XGBClassifier(objective='multi:softmax', n_estimators=200, learning_rate=0.05, max_depth=20, nthread=4, subsample=0.7, colsample_bytree=0.85, seed=2471)
      numberEpochs = 10
    elif source == '4':
      clf = xgb.XGBClassifier(objective='multi:softmax', n_estimators=200, learning_rate=0.05, max_depth=20, nthread=4, subsample=0.7, colsample_bytree=0.85, seed=2471)
      numberEpochs = 30

    clf.fit(X_train, Y_train, early_stopping_rounds=20, eval_metric='mlogloss', eval_set=[(X_valid, Y_valid)])

    predictValidY = clf.predict_proba(X_valid)
    #score = metrics.log_loss(to_categorical(Y_valid, 10), predictValidY)
    print('Fold ' + str(num_fold) + ' done')

    for i in range(len(test_index)):
      yfull_train[test_index[i]] = predictValidY[i]

    test_prediction = clf.predict_proba(xTest)
    yfull_test.append(test_prediction)

    if gcEnabled:
      gc.collect()

    num_fold += 1

  score = metrics.log_loss(to_categorical(y, 10), dict_to_list(yfull_train))
  print('Final log loss ' + str(score))
  test_res = merge_several_folds_mean(yfull_test, nfolds)
  predictions = pd.DataFrame(test_res, index=files)
  predictions.index.name = 'img'
  predictions.columns = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
  predictions.to_csv(predictionsFile)

run_cross_validation(10)