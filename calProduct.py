# calculate the product probability and output the transprob 
# import calTrans as ct
import numpy as np
class Product(object):
	def __init__(self, inputFileRaw, states, symbols, inputFileCenter, inputFileRle):
		self.states = states
		self.symbols = symbols
		self.inputFileRaw = inputFileRaw
		self.inputFileCenter = inputFileCenter
		self.inputFileRle = inputFileRle

	def getCenterPoint(self):
		dictArr = []
		file = open(self.inputFileCenter)
		for line in file:
			arr = line.split(':')
			arr[0] = int(arr[0].strip(' '))
			arr[1] = float(arr[1].strip(' []\n'))
			dictArr.append(arr)
		centerDict = dict(dictArr)
		return centerDict
	#cal the cluster's variance to define the norm's parameters
	def getVarianceByCluster(self):
		rleArr = []
		dictArrSym = []
		for i in self.symbols:
			arr = [i , []]
			dictArrSym.append(arr)
			arr = []
		valueDict = dict(dictArrSym)
		dictArrSym = []
		#meansDict = self.getCenterPoint()
		rleFile = open(self.inputFileRle)
		for rleLine in rleFile.readlines():
			rleArr = rleLine.strip('\n').split(',')
			rawFile = open(self.inputFileRaw)
			for rawLine in rawFile.readlines()[(int(rleArr[0])/10):(int(rleArr[1])/10)]:
				rawArr = rawLine.split(',')
				valueDict[int(rleArr[2])].append(int(rawArr[2]))
			rleArr = []
		for i in self.symbols:
			values = valueDict[i]
			maxValue = max(values)
			minValue = min(values)
			stand = np.std(values)
			means = np.mean(values)
			arr2 = [i , [means, stand, maxValue, minValue]]
			dictArrSym.append(arr2)
			arr2 = []
		varDict = dict(dictArrSym)
		return varDict
	#cal the pattern's means to define the Product Prob
	def getMeansBySymbol(self):
		rleArr = []
		rawArr = []
		valueArr = []
		valuesArr = []
		rleFile = open(self.inputFileRle)
		for rleLine in rleFile.readlines():
			rleArr = rleLine.strip('\n').split(',')
			rawFile = open(self.inputFileRaw)
			for rawLine in rawFile.readlines()[(int(rleArr[0])/10):int(rleArr[1])/10]:
				rawArr = rawLine.split(',')
				valueArr.append(int(rawArr[2]))
			rawArr = []
			means = np.mean(valueArr)
			valuesArr.append(means)
			valueArr = []
		return valuesArr
ss = Product('/home/hao/newworkspaces/StateFinder-master/StateFinder/input-example/data.csv', [0,1,2,3,4,5,6], (0 , 1 , 2 , 3 , 4 , 5 , 6), '/home/hao/newworkspaces/StateFinder-master/Spclust/cent.log', '/home/hao/newworkspaces/StateFinder-master/data-rle.csv')
print ss.getMeansBySymbol()
print ss.getVarianceByCluster()