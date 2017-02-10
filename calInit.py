# coding: utf-8
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
def uniqueIndex(L, e):
	return [ i for (i, j) in enumerate(L) if j == e]

def normlizeTrans(transList):
	length = len(transList)
	count = map(sum,transList)
	for i in range(0, length):
		for j in range(0, length):
			transList[i][j] = transList[i][j]/count[i]
	return transList

def kde_sklearn(x, x_grid, bandwidth, **kwargs):
	kde = KernelDensity(bandwidth = bandwidth, **kwargs)
	# fit Fit the Kernel Density model on the data.
	kde.fit(x[:, np.newaxis]) #newaxis 可以新增维度, array([1., 2., 3.])  >>> a[:,np.newaxis] array([[ 1.], [ 2.], [ 3.]])
	pdf = kde.score_samples(x_grid[:, np.newaxis])
	return np.exp(pdf)

# def function_kde(x, xj, n, h):
# 	func = 1/(n*h*(2*np.pi)**0.5)*np.e**(-(x-xj)**2/(2*h**2))
# 	return func
class DataInit(object):

	def __init__(self, inputFileRaw, states, symbols, listTest, inputFileRle, inputFileSymbol):
		self.inputFileRaw = inputFileRaw
		self.inputFileRle = inputFileRle
		self.inputFileSymbol = inputFileSymbol
		self.states = states
		self.symbols = symbols
		self.listTest = listTest
		self.rleTest = []
		symbolTest = []
		symbolTmp = []
		stateTmp = []
		# with open(self.inputFileRle) as f:
		# 	reader = csv.reader(f)
		# 	for row in reader:
		# 		self.listTest.append(int(row[2]))
		with open(self.inputFileRaw) as f:
			reader = csv.reader(f)
			for row in reader:
				self.listTest.append(int(row[2]))

		with open(self.inputFileSymbol) as f:
			reader = csv.reader(f)
			for row in reader:
				symbolTest.append(int(row[2]))	

		with open(self.inputFileRle) as f:
			reader = csv.reader(f)
			for row in reader:
				self.rleTest.append(int(row[2]))
		# load file
		symbolTmp = np.unique(symbolTest)
		symbolTmp = [int(symbolTmp[i]) for i in symbolTmp]
		for index in range(len(symbolTmp)):
			stateTmp.append(symbolTmp[index])

		self.states = tuple(stateTmp) #change into init states	
		self.symbols = tuple(symbolTmp) #change into patterns
	
	def symbolFinder(self):
		return self.symbols
	
	def stateFinder(self):
		return self.states
	
	def listFinder(self):
		return self.listTest

	def transProb(self):
		rleTest = self.rleTest
		states = self.states
		statesLen = len(states)
		transList = [[0]*statesLen for row in range(statesLen)] #init transMatrix 2 dim
		indexList = []
		roundCount = 0 #each turn to count the transactions number
		count = float(len(rleTest)) #change mod to float 
		maxNum = int(max(rleTest))
		minNum = int(min(rleTest))
		for i in range(minNum, maxNum + 1):
			indexList = uniqueIndex(rleTest, i)
			for j in range(minNum, maxNum + 1):
				for k in indexList: #change into if else
					#params by RLE sequence
					if k == (len(rleTest)-1):
						print "the last element in list"
					#j在K的后面一位
					elif j == rleTest[k+1] :
						roundCount += 1
				transList[i][j] = roundCount / count
				roundCount = 0
		transList = normlizeTrans(transList)
		return transList

	def calStartProb(self):
		startTestProb = {}
		for index in range(len(self.states)):
			prob = 1.0/len(self.states)
			startTestProb[self.states[index]] = prob
			# if index == 0:
			# 	startTestProb[self.states[index]] = 1
			# else:
			# 	startTestProb[self.states[index]] = 0
		# init pi start_probaility
		startProb = dict(sorted(startTestProb.items(), key=lambda e:e[0], reverse=False))#sorted return a list not a dictionary and use dict we got a dictionary
		return startProb
	# def getCenterPoint(self):
	# 	dictArr = []
	# 	file = open(self.inputFileCenter)
	# 	for line in file:
	# 		arr = line.split(':')
	# 		arr[0] = int(arr[0].strip(' '))
	# 		arr[1] = float(arr[1].strip(' []\n'))
	# 		dictArr.append(arr)
	# 	centerDict = dict(dictArr)
	# 	return centerDict

	#cal the cluster's variance to define the norm's parameters
	def getVarianceByCluster(self):
		rleArr = []
		valueArr = []
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
				# valueArr.append(int(rawArr[2]))#每一个pattern中点值的个数
				# patternMeans = np.mean(valueArr)#此步计算每一个pattern的均值
				# valueDict[int(rleArr[2])].append(patternMeans)
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
		countsArr = []
		rleFile = open(self.inputFileRle)
		for rleLine in rleFile.readlines():
			rleArr = rleLine.strip('\n').split(',')
			rawFile = open(self.inputFileRaw)
			#10 params 
			for rawLine in rawFile.readlines()[(int(rleArr[0])/10):int(rleArr[1])/10]:
				rawArr = rawLine.split(',')
				valueArr.append(int(rawArr[2]))
			rawArr = []
			#count = len(valueArr)
			#summ = sum(valueArr)
			means = np.mean(valueArr)
			valuesArr.append(means)
			valueArr = []
		return valuesArr

	def calDistrubitionByMeans(self, state):
		disDict = {}
		valueArr = []
		means = 0
		fileRle = open(self.inputFileRle)
		states = self.states
		for i in states:
			disDict[i] = []
		for rleLine in fileRle.readlines():
			rleArr = rleLine.strip('\n').split(',')
			rawFile = open(self.inputFileRaw)
			for i in states:
				if int(rleArr[2]) == i:
					for rawLine in rawFile.readlines()[(int(rleArr[0])/10):(int(rleArr[1])/10)]:
						rawArr = rawLine.split(',')
						valueArr.append(int(rawArr[2]))
						#disDict[i].append(int(rawArr[2]))
					means = np.mean(valueArr)
					disDict[i].append(means)
					valueArr = []
		zero = disDict[state]
		maxNum = max(zero)
		minNum = min(zero)
		x_grid = np.linspace(minNum,maxNum,100)
		np.random.seed(0)
		x = np.array(zero)

		# cal the bandwidth for KDE
		std = np.std(zero)
		n = len(zero)
		h = 1.06*std/(n**0.2)
		# cal the bandwidth for KDE 
		
		# list the function of KDE
		# total_func = 0
		# for i in range(0, n):
		# 	total_func = total_func + function_kde()
		# list the function of KDE
		
		# #display the figure
		# fig,ax = plt.subplots(1,1, sharey=True)
		# fig.subplots_adjust(wspace=0)
		# pdf = kde_sklearn(x,x_grid,bandwidth=h) # 0:80~90 1:
		# ax.plot(x_grid,pdf, color='green',alpha=0.5,lw=1)
		# ax.hist(x, 40, fc='black',histtype='stepfilled',alpha=0.2, normed=True)
		# ax.set_xlim(minNum,maxNum)
		# plt.show()
		# #display the figure
		return zero, h, n

	def calDistrubitionByDistance(self, state):
		disDict = {}
		valueArr = []
		patternMeansDict = {}
		patternMeans = 0
		meansArr = []
		zero = []
		fileRle = open(self.inputFileRle)
		states = self.states
		for i in states:
			disDict[i] = []
			patternMeansDict[i] = []
		for rleLine in fileRle.readlines():
			rleArr = rleLine.strip('\n').split(',')
			rawFile = open(self.inputFileRaw)
			for i in states:
				if int(rleArr[2]) == i:
					for rawLine in rawFile.readlines()[(int(rleArr[0])/10):(int(rleArr[1])/10)]:
						rawArr = rawLine.split(',')
						valueArr.append(int(rawArr[2]))
						disDict[i].append(int(rawArr[2]))
					patternMeans = np.mean(valueArr)
					patternMeansDict[i].append(patternMeans)
					#disDict[i].append(valueArr)
					valueArr = []
		# print patternMeansDict
		# print disDict
		stateMeans = np.mean(disDict[state])
		for i in range(len(patternMeansDict[state])):
			zero.append(math.fabs(stateMeans - patternMeansDict[state][i]))
		#zero = disDict[state]
		maxNum = max(zero)
		minNum = min(zero)
		x_grid = np.linspace(minNum,maxNum,100)
		np.random.seed(0)
		x = np.array(zero)

		# cal the bandwidth for KDE
		std = np.std(zero)
		n = len(zero)
		h = 1.06*std/(n**0.2)
		# cal the bandwidth for KDE 
		
		# list the function of KDE
		# total_func = 0
		# for i in range(0, n):
		# 	total_func = total_func + function_kde()
		# list the function of KDE
		
		# #display the figure
		# fig,ax = plt.subplots(1,1, sharey=True)
		# fig.subplots_adjust(wspace=0)
		# pdf = kde_sklearn(x,x_grid,bandwidth=h) # 0:80~90 1:
		# ax.plot(x_grid,pdf, color='green',alpha=0.5,lw=1)
		# ax.hist(x, 40, fc='black',histtype='stepfilled',alpha=0.2, normed=True)
		# ax.set_xlim(minNum,maxNum)
		# plt.show()
		# #display the figure
		return zero, h, n

	def displayDistrub(self,state):
		disDict = {}
		valueArr = []
		valuesArr = []
		fileRle = open(self.inputFileRle)
		states = self.states
		for i in states:
			disDict[i] = []
		for rleLine in fileRle.readlines():
			rleArr = rleLine.strip('\n').split(',')
			rawFile = open(self.inputFileRaw)
			for i in states:
				if int(rleArr[2]) == i:
					for rawLine in rawFile.readlines()[(int(rleArr[0])/10):(int(rleArr[1])/10)]:
						rawArr = rawLine.split(',')
						valueArr.append(int(rawArr[2]))
						#disDict[i].append(int(rawArr[2]))
					means = np.mean(valueArr)
					disDict[i].append(means)
					valueArr = []
		zero = disDict[state]
		maxNum = max(zero)
		minNum = min(zero)
		x_grid = np.linspace(minNum,maxNum,10)
		np.random.seed(0)
		x = np.array(zero)
		print disDict
		# cal the bandwidth for KDE
		std = np.std(zero)
		n = len(zero)
		h = 1.06*std/(n**0.2)
		# cal the bandwidth for KDE 
		
		#display the figure
		fig,ax = plt.subplots(1,1, sharey=True)
		fig.subplots_adjust(wspace=0)
		pdf = kde_sklearn(x,x_grid,bandwidth=h) # 0:80~90 1:
		ax.plot(x_grid,pdf, color='green',alpha=0.5,lw=1)
		ax.hist(x, 40, fc='black',histtype='stepfilled',alpha=0.2, normed=True)
		ax.set_xlim(minNum,maxNum)
		plt.show()
		#display the figure

	def showDistrubitionByDistance(self, state):
		disDict = {}
		valueArr = []
		patternMeansDict = {}
		patternMeans = 0
		meansArr = []
		zero = []
		fileRle = open(self.inputFileRle)
		states = self.states
		for i in states:
			disDict[i] = []
			patternMeansDict[i] = []
		for rleLine in fileRle.readlines():
			rleArr = rleLine.strip('\n').split(',')
			rawFile = open(self.inputFileRaw)
			for i in states:
				if int(rleArr[2]) == i:
					for rawLine in rawFile.readlines()[(int(rleArr[0])/10):(int(rleArr[1])/10)]:
						rawArr = rawLine.split(',')
						valueArr.append(int(rawArr[2]))
						disDict[i].append(int(rawArr[2]))
					patternMeans = np.mean(valueArr)
					patternMeansDict[i].append(patternMeans)
					#disDict[i].append(valueArr)
					valueArr = []
		print patternMeansDict
		print disDict
		stateMeans = np.mean(disDict[state])
		print stateMeans
		for i in range(len(patternMeansDict[state])):
			zero.append(math.fabs(stateMeans - patternMeansDict[state][i]))
		print zero
		maxNum = max(zero)
		minNum = min(zero)
		x_grid = np.linspace(minNum,maxNum,100)
		np.random.seed(0)
		x = np.array(zero)

		# cal the bandwidth for KDE
		std = np.std(zero)
		n = len(zero)
		h = 1.06*std/(n**0.2)
		# cal the bandwidth for KDE 
		
		# list the function of KDE
		# total_func = 0
		# for i in range(0, n):
		# 	total_func = total_func + function_kde()
		# list the function of KDE
		
		# #display the figure
		fig,ax = plt.subplots(1,1, sharey=True)
		fig.subplots_adjust(wspace=0)
		pdf = kde_sklearn(x,x_grid,bandwidth=h) # 0:80~90 1:
		ax.plot(x_grid,pdf, color='green',alpha=0.5,lw=1)
		ax.hist(x, 40, fc='black',histtype='stepfilled',alpha=0.2, normed=True)
		ax.set_xlim(minNum,maxNum)
		plt.show()
		# #display the figure
		return zero, h, n


	def calRawDataForSymbol(self):
			rawDataDict = {}
			fileRle = open(self.inputFileRle)
			symbols = self.symbols
			for i in symbols:
				rawDataDict[i] = []
			for rleLine in fileRle.readlines():
				rleArr = rleLine.strip('\n').split(',')
				rawFile = open(self.inputFileRaw)
				for i in symbols:
					if int(rleArr[2]) == i:
						for rawLine in rawFile.readlines()[(int(rleArr[0])/10):(int(rleArr[1])/10)]:
							rawArr = rawLine.split(',')
							rawDataDict[i].append(int(rawArr[2]))
			return rawDataDict
# ss = DataInit('/home/hao/newworkspaces/StateFinder-master/StateFinder/input-example/data.csv', (), (), [], '/home/hao/newworkspaces/StateFinder-master/data35-rle.csv', '/home/hao/newworkspaces/StateFinder-master/data35-symbol.csv')
# print ss.getMeansBySymbol()
# print ss.getVarianceByCluster()
# print ss.stateFinder() #state string
# print ss.symbolFinder()
# print ss.listFinder() #list patterns string
# # print ss.calRawDataForSymbol()
# #print ss.transProb() #aij
# # # # print ss.calStartProb()
# # # # print ss.calStartProb()
# #ss.displayDistrub(2)
# # ss.showDistrubitionByDistance(4)