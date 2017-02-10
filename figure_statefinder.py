import matplotlib.pyplot as plt
import numpy as np
x1 = np.loadtxt("/home/hao/newworkspaces/StateFinder-master/data35-statefinder.csv", delimiter = ",", usecols = (0,), dtype = int)
x2 = np.loadtxt("/home/hao/newworkspaces/StateFinder-master/data35-statefinder.csv", delimiter = ",", usecols = (1,), dtype = int)
#y2 = np.loadtxt("/home/hao/datasets/REDD/test.dat", delimiter = " ", usecols = (1,), dtype = float)
y2 = np.loadtxt("/home/hao/newworkspaces/StateFinder-master/data35-statefinder.csv", delimiter = ",", usecols = (2,), dtype = int)
for i in xrange(0, len(y2)):
	tmp = 0 - y2[i]
	y2[i] = tmp
y1 = y2.tolist()
y1.insert(0,0)
del y1[-1]

a1 = np.loadtxt("/home/hao/newworkspaces/StateFinder-master/data35-rle.csv", delimiter = ",", usecols = (0,), dtype = int)
a2 = np.loadtxt("/home/hao/newworkspaces/StateFinder-master/data35-rle.csv", delimiter = ",", usecols = (1,), dtype = int)
z2 = np.loadtxt("/home/hao/newworkspaces/StateFinder-master/data35-rle.csv", delimiter = ",", usecols = (2,), dtype = int)
z1 = z2.tolist()
z1.insert(0,0)
del z1[-1]
plt.plot([x1,x2], [y2,y2],'b-',[x1,x1],[y1,y2],'r--',[a1,a2],[z2,z2],'g-',[a1,a1],[z1,z2],'r--')
plt.xlabel('time')
plt.ylabel('symbol')
plt.axis([0,24000,-6,6])
plt.show()
