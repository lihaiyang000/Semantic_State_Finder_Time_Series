from calInit import DataInit
import numpy as np
import matplotlib.pyplot as plt
import csv
x1 = np.loadtxt("/home/hao/py-work/PythonHMM/data.csv", delimiter = ",", usecols = (0,), dtype = int)
x2 = np.loadtxt("/home/hao/py-work/PythonHMM/data.csv", delimiter = ",", usecols = (1,), dtype = int)
#y2 = np.loadtxt("/home/hao/datasets/REDD/test.dat", delimiter = " ", usecols = (1,), dtype = float)
y2 = np.loadtxt("/home/hao/py-work/PythonHMM/data.csv", delimiter = ",", usecols = (2,), dtype = int)
y1 = y2.tolist()
y1.insert(0,0)
del y1[-1]
plt.plot([x1,x2], [y2,y2],'b-',[x1,x1],[y1,y2],'r--')
plt.xlabel('time')
plt.ylabel('value')
plt.axis([0,24000,-0.5,7000])
fileRle = open("/home/hao/newworkspaces/StateFinder-master/data35-rle.csv")
valueArr = []
rleArr = []
rawArr = []
for line in fileRle.readlines():
	rleArr = line.strip('\n').split(',')
	mean = (int(rleArr[1]) - int(rleArr[0]))/2 + int(rleArr[0])
	fileRaw = open("/home/hao/py-work/PythonHMM/data.csv")
	for rawLine in fileRaw.readlines()[(int(rleArr[0])/10):(int(rleArr[1])/10)]:
		rawArr = rawLine.strip('\n').split(',')
		valueArr.append(int(rawArr[2]))
	print valueArr
	rawMax = max(valueArr)
	axisY = rawMax + 200
	plt.annotate(str(rleArr[2]),xy=(mean,axisY),xytext=(mean,axisY))
	valueArr = []
# plt.annotate('1',xy=(70,500),xytext=(70,500))
# plt.annotate('2',xy=(175,1500),xytext=(175,1500))
plt.show()