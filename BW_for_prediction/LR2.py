print(__doc__)
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import csv


trainX = []
fileX = open('./trainX.log', 'r')
while True:
    s = fileX.readline()
    if s == '':
        break
    tem = []
    for x in s.strip().strip('[').strip(']').split(','):
        tem.append(float(x))
    trainX.append(tem)
trainX = np.array(trainX).T

# trainX3 = trainX[:3].T
# trainX6 = trainX[3:6].T
# trainX9 = trainX[6:].T
#
# trainX = []
# for i in range(len(trainX3)):
#     tem = []
#     tem.append(np.max(trainX3[i]))
#     tem.append(np.max(trainX6[i]))
#     tem.append(np.max(trainX9[i]))
#     trainX.append(tem)
# print(np.array(trainX))
# trainX = np.array(trainX)
# trainX = np.array(trainX)
# print(trainX)


test = []
fileY = open('./test.log', 'r')
while True:
    s = fileY.readline()
    if s == '':
        break
    tem = []
    for x in s.strip().strip('[').strip(']').split(','):
        if x.strip() != '':
            tem.append(float(x.strip()))

    test.append(tem)
test = np.array(test).T
# test3 = test[:3].T
# test6 = test[3:6].T
# test9 = test[6:].T
# test = []
# for i in range(len(test3)):
#     tem = []
#     tem.append(np.max(test3[i]))
#     tem.append(np.max(test6[i]))
#     tem.append(np.max(test9[i]))
#     test.append(tem)
# test = np.array(test)
# print(test)
trainY = []
with open('./trainY.log', 'r') as f:
    for y in f.read().strip().strip('[').strip(']').split(','):
        trainY.append(float(y))
trainY = np.array(trainY)
print(trainX)
# print(len(trainY))
print(trainY)
print(test)

regr = linear_model.LinearRegression()
print(len(trainX))
print(len(trainY))
regr.fit(trainX, trainY)
print(regr.predict(test))

