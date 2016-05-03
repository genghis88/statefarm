import pandas as pd
from os import path
from os import listdir
from PIL import Image
import pickle
import sys
import numpy as np
import xgboost as xgb

modelFile = sys.argv[1]
predictionsFile = sys.argv[2]

testDir = 'data/test_normalized'

imageSize = 1200
batchSize = 100

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
xTest = xTest.reshape(xTest.shape[0], 1, 40, 30).astype('float32')
print(xTest.shape)

#xTest /= xTest.std(axis = None)
#xTest -= xTest.mean()

model = open(modelFile,'rb')
net = pickle.load(model)

testY = net.predict(xTest)
print(testY)
print(testY.shape)
predictY = net.predict_proba(xTest)
print(predictY)
print(predictY.shape)
#predictY = np.reshape(predictY, (len(files), 1))

def extend(val):
  a = np.zeros(10)
  a[val] = 1
  return a

#predictions = np.zeros((len(files), 10), dtype='float32')
#predictY = np.apply_along_axis(extend, axis=1, arr=predictY)
#print(predictY)
#print(predictY.shape)

predictions = pd.DataFrame(predictY, index=files)
predictions.index.name = 'img'

predictions.columns = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

predictions.to_csv(predictionsFile)
