import matplotlib.pyplot as plt
import numpy as np
x1 = np.loadtxt("/home/hao/newworkspaces/StateFinder-master/StateFinder/input-example/data.csv", delimiter = ",", usecols = (0,), dtype = int)
x2 = np.loadtxt("/home/hao/newworkspaces/StateFinder-master/StateFinder/input-example/data.csv", delimiter = ",", usecols = (1,), dtype = int)
#y2 = np.loadtxt("/home/hao/datasets/REDD/test.dat", delimiter = " ", usecols = (1,), dtype = float)
y2 = np.loadtxt("/home/hao/newworkspaces/StateFinder-master/StateFinder/input-example/data.csv", delimiter = ",", usecols = (2,), dtype = int)
y1 = y2.tolist()
y1.insert(0,0)
del y1[-1]
print y1
plt.plot(x1, y2,'b-')
plt.xlabel('time')
plt.ylabel('symbol')
plt.axis([0,24000,-0.5,10000])
plt.show()