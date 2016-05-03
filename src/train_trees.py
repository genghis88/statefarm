import pandas as pd
from os import path
from PIL import Image
import pickle
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn import cross_validation
from keras.utils.np_utils import to_categorical
import xgboost as xgb

trainFile = sys.argv[1]
pickleFile = sys.argv[2]

trainingDir = 'data/train_normalized'

trainingLabels = pd.read_csv(trainFile)
print(trainingLabels.shape)

#print(trainingLabels[trainingLabels['subject'] == 'p002'])

imageSize = 1200
batchSize = 100
xTrain = np.zeros((trainingLabels.shape[0], imageSize), dtype='float32')
for index, row in trainingLabels.iterrows():
  fileName = path.join(trainingDir, row['classname'] + row['img'])
  #print(fileName)
  im = Image.open(fileName)
  xTrain[index, :] = np.reshape(im, imageSize)
  im.close()

xTrain /= 255
#xTrain = xTrain.reshape(xTrain.shape[0], 1, 40, 30).astype('float32')
#print(xTrain.shape)

#xTrain /= xTrain.std(axis = None)
#xTrain -= xTrain.mean()

y = np.array([int(x[-1:]) for x in trainingLabels['classname']]).astype('int32')
#y = to_categorical(y, 10)
print(y.shape)

x_fit, x_eval, y_fit, y_eval = cross_validation.train_test_split(xTrain, y, test_size=0.2)

clf = xgb.XGBClassifier(objective='multi:softmax', n_estimators=200, learning_rate=0.05, max_depth=20, nthread=4, subsample=0.7, colsample_bytree=0.85, seed=2471)

clf.fit(x_fit, y_fit, early_stopping_rounds=20, eval_metric='mlogloss', eval_set=[(x_eval, y_eval)])

clf.fit(xTrain, y)
predictY = clf.predict_proba(xTrain)

from sklearn import metrics

y = to_categorical(y, 10)
print(metrics.coverage_error(y, predictY))

with open(pickleFile,'wb') as f:
  sys.setrecursionlimit(20000)
  pickle.dump(clf, f)
