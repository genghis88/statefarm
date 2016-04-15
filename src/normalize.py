import pandas as pd
from os import path
from os import listdir
from PIL import Image
import pickle
import cv2
import sys
import numpy as np

trainFile = sys.argv[1]
trainingOutputDir = sys.argv[2]
testingOutputDir = sys.argv[3]

trainingDir = 'data/train'
testDir = 'data/test'

trainingLabels = pd.read_csv(trainFile)
print(trainingLabels.shape)

imageSize = (40, 30)
batchSize = 100
for index, row in trainingLabels.iterrows():
  fileName = path.join(trainingDir, row['classname'], row['img'])
  #im = Image.open(fileName)
  #im = im.resize(imageSize, Image.ANTIALIAS)
  #im = im.convert('L')
  im = cv2.imread(fileName, 0)
  im = cv2.resize(im, imageSize)
  newFileName = path.join(trainingOutputDir, row['classname'] + row['img'])
  cv2.imwrite(newFileName, im)

files = [ f for f in listdir(testDir) if path.isfile(path.join(testDir,f)) ]

for fName in files:
  fileName = path.join(testDir,fName)
  im = cv2.imread(fileName, 0)
  im = cv2.resize(im, imageSize)
  newFileName = path.join(testingOutputDir, fName)
  cv2.imwrite(newFileName, im)
