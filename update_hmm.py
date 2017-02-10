# -*- coding: utf-8 -*-
# In this round, we update hmm from the patterns.
# Each pattern is respond to a state, each state is the result of clustering patterns, so the new patterns cluster can be the state's parameters.
# Then, update the state's means and std,study the new normal disturbition, and update the trans_prob, init_prob according to the states' total number
import numpy as np

def uniqueIndex(L, e):
	return [ i for (i, j) in enumerate(L) if j == e]

def normlizeTrans(transList):
	length = len(transList)
	count = map(sum,transList)
	for i in range(0, length):
		for j in range(0, length):
			transList[i][j] = transList[i][j]/count[i]
	return transList

class UpdateHmm(object):
	def __init__(self, result):
		self.resultFile = result
		stateTotalList = []
		resultFile = open(self.resultFile)
		for resultLine in resultFile.readlines():
			resultArr = resultLine.strip('\n').split(',')
			stateTotalList.append(int(resultArr[3]))
		stateList = np.unique(stateTotalList)
		self.stateTotalList = stateTotalList
		self.stateList = stateList

	#return a dictionary for pattern and state
	def findPattern(self):
		patternDict = {}
		for i in self.stateList:
			patternDict[int(self.stateList[i])] = []
		resultFile = open(self.resultFile)
		for resultLine in resultFile.readlines():
			resultArr = resultLine.strip('\n').split(',')
			patternDict[int(resultArr[3])].append(float(resultArr[2]))
		return patternDict

	def findAllPattern(self):
		patternArr = []
		resultFile = open(self.resultFile)
		for resultLine in resultFile.readlines():
			resultArr = resultLine.strip('\n').split(',')
			flo = float(resultArr[2])
			resultInt = int(flo)
			patternArr.append(resultInt)
		return patternArr
	#return state's statistic informations
	def getVarianceByCluster(self, patternDict):
		statisticDict = {}
		for i in self.stateList:
			means = np.mean(patternDict[int(self.stateList[i])])
			std = np.std(patternDict[int(self.stateList[i])])
			if std == 0:
				std = 1
			maxValue = max(patternDict[int(self.stateList[i])])
			minValue = min(patternDict[int(self.stateList[i])])
			maxValue = maxValue +500
			minValue = minValue - 500
			statisticDict[int(self.stateList[i])] = [means, std, maxValue, minValue]
		return statisticDict

	def calStartProb(self):
		startProb = {}
		for index in range(len(self.stateList)):
			prob = 1.0/len(self.stateList)
			startProb[self.stateList[index]] = prob
		return startProb

	#return transprob
	def calTransProb(self):
		statesLen = len(self.stateList)
		transList = [[0]*statesLen for row in range(statesLen)] #init transMatrix 2 dim
		indexList = []
		roundCount = 0 #each turn to count the transactions number
		count = float(len(self.stateTotalList)) #change mod to float 
		for i in self.stateList:
			indexList = uniqueIndex(self.stateTotalList, i)
			for j in self.stateList:
				for k in indexList: #change into if else
					#199 params
					if k == (len(self.stateTotalList)-1):
						print "the last element in list"
					#j在K的后面一位
					elif j == self.stateTotalList[k+1] :
						roundCount += 1
				transList[i][j] = roundCount / count
				roundCount = 0
		transList = normlizeTrans(transList)
		return transList

# test = UpdateHmm('./test_resultraw2.log')
# patterns = test.findPattern()
# meansStd = test.getVarianceByCluster(patterns)
# print patterns
# print meansStd
# print test.stateTotalList
# print tuple(test.stateList) #a1
# print tuple(test.stateList) #b1
# print test.calStartProb() #c1
# print test.findAllPattern() #e1
# print test.calTransProb() #d1