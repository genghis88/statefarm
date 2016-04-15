import pandas as pd
from os import path
from PIL import Image
import pickle
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxoutDense, Activation, Convolution2D, MaxPooling2D, Flatten, AveragePooling2D
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam, Adamax
from keras.layers.advanced_activations import ELU, PReLU
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn import cross_validation

trainFile = sys.argv[1]
pickleFile = sys.argv[2]

trainingDir = 'data/train_normalized'

trainingLabels = pd.read_csv(trainFile)
print(trainingLabels.shape)

#print(trainingLabels[trainingLabels['subject'] == 'p002'])

imageSize = 1200
batchSize = 100
xTrain = np.zeros((trainingLabels.shape[0], imageSize))
for index, row in trainingLabels.iterrows():
  fileName = path.join(trainingDir, row['classname'] + row['img'])
  #print(fileName)
  im = Image.open(fileName)
  xTrain[index, :] = np.reshape(im, imageSize)
  im.close()

xTrain /= 255
xTrain = xTrain.reshape(xTrain.shape[0], 1, 40, 30).astype('float32')
print(xTrain.shape)

#xTrain /= xTrain.std(axis = None)
#xTrain -= xTrain.mean()

y = np.array([int(x[-1:]) for x in trainingLabels['classname']]).astype('int32')
y = to_categorical(y, 10)
print(y.shape)

#scaler = StandardScaler()
#scaler.fit(xTrain)
#xTrain = scaler.transform(xTrain)

#xTrain = normalize(xTrain, axis=0)

net = Sequential()
net.add(Convolution2D(32, 3, 3, input_shape=(xTrain.shape[1], xTrain.shape[2], xTrain.shape[3]), init='glorot_uniform'))
net.add(ELU())
net.add(MaxPooling2D(pool_size=(2,2)))
net.add(Dropout(0.2))
net.add(Convolution2D(64, 4, 3, init='glorot_uniform'))
net.add(Activation('tanh'))
net.add(MaxPooling2D(pool_size=(2,2)))
net.add(Dropout(0.2))
net.add(Convolution2D(128, 3, 3, init='glorot_uniform'))
net.add(Activation('tanh'))
net.add(MaxPooling2D(pool_size=(2,2)))
net.add(Dropout(0.2))
net.add(Convolution2D(256, 3, 2, init='glorot_uniform'))
net.add(Activation('tanh'))
net.add(Flatten())
net.add(Dense(1024, activation='tanh'))
net.add(Dense(10, activation='softmax'))
#net.add(Dense(1, activation='relu'))

#optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-10)

net.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

x_fit, x_eval, y_fit, y_eval= cross_validation.train_test_split(xTrain, y, test_size=0.2)

net.fit(x_fit, y_fit,
          nb_epoch=30,
          batch_size=batchSize,
          verbose=1,
          validation_data=(x_eval, y_eval))

with open(pickleFile,'wb') as f:
  sys.setrecursionlimit(20000)
  pickle.dump(net, f)
