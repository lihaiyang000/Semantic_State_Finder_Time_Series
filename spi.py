import numpy as np
import csv
import math

def symbolList(symbolFile):
	symbolArr = []
	symbolFile = open(symbolFile)
	for line in symbolFile.readlines():
		tempArr = line.strip('\n').split(',')
		symbolArr.append(int(tempArr[2]))
	return symbolArr

def calConnectCluster(cluster1, cluster2, symbol):
	symbolArr = symbol
	count = 0
	# for line in symbolFile.readlines():
	# 	tempArr = line.strip('\n').split(',')
	# 	symbolArr.append(int(tempArr[2]))
	for i in xrange(len(symbolArr)-1):
		if (symbolArr[i] == cluster1 and symbolArr[i+1] == cluster2):
			count += 1
	return count

def diameter(cluster, symbolFile, rawFile):
	valueArr = []
	symbol = open(symbolFile)
	for line in symbol.readlines():
		tempArr = line.strip('\n').split(',')
		if int(tempArr[2]) == cluster:
			raw = open(rawFile)
			for rawLine in raw.readlines():
				rawArr = rawLine.strip('\n').split(',')
				if (int(rawArr[0]) == int(tempArr[0]) and int(rawArr[1]) == int(tempArr[1])):
					valueArr.append(int(rawArr[2]))
	maxNum = max(valueArr)
	minNum = min(valueArr)
	diam = maxNum - minNum
	return diam

def distanceConnect(cluster1,cluster2,symbol,rawFile):
	symbolArr = symbol
	rawArr = []
	distArr = []
	distanceCount = 0
	# symbol = open(symbolFile)
	# for line in symbol.readlines():
	# 	tempArr = line.strip('\n').split(',')
	# 	symbolArr.append(int(tempArr[2]))
	raw = open(rawFile)
	for rawLine in raw.readlines():
		tmpRaw = rawLine.strip('\n').split(',')
		rawArr.append(int(tmpRaw[2]))
	for i in xrange(len(symbolArr)-1):
		if(symbolArr[i] == cluster1 and symbolArr[i+1] == cluster2):
			dis = math.fabs(rawArr[i] - rawArr[i+1])
			distArr.append(dis)
	print distArr
	for k in xrange(len(distArr)):
		distanceCount += distArr[k]
	return distanceCount

def avgConnLen(cluster1, cluster2, symbol, rawFile):
	disCount = distanceConnect(cluster1, cluster2, symbol, rawFile)
	conn = calConnectCluster(cluster1, cluster2, symbol)
	if conn != 0:
		avgConn = float(disCount)/conn
	else:
		avgConn = 0
	return avgConn

# symbolFile = '/home/hao/newworkspaces/StateFinder-master/Spclust/channel-symbol.csv'
# rawFile = '/home/hao/datasets/REDD/channel_11.csv'
# symbolArr = symbolList('/home/hao/newworkspaces/StateFinder-master/Spclust/channel-symbol.csv')
symbolFile = '/home/hao/newworkspaces/StateFinder-master/result_30/30_30_symbol.csv'
rawFile = '/home/hao/newworkspaces/StateFinder-master/StateFinder/input-example/data.csv'
symbolArr = symbolList('/home/hao/newworkspaces/StateFinder-master/result_30/30_30_symbol.csv')
uniSymbol = np.unique(symbolArr) #np.array
intraCoeff = 0
interCoeff = 0
# compute the intra
for i in xrange(len(uniSymbol)):
	count = calConnectCluster(uniSymbol[i],uniSymbol[i],symbolArr)
	diam = diameter(uniSymbol[i],symbolFile,rawFile)
	intraCoeff = intraCoeff + float(count)/diam
print intraCoeff

# compute the inter
for i in xrange(len(uniSymbol)):
	reuniSymbol = np.delete(uniSymbol, i)
	for j in xrange(len(reuniSymbol)):
		avgConn = avgConnLen(uniSymbol[i], reuniSymbol[j], symbolArr, rawFile)
		interConn = calConnectCluster(uniSymbol[i], reuniSymbol[j], symbolArr)
		if avgConn != 0:
			print uniSymbol[i] , reuniSymbol[j]
			print "avgConn is " + str(avgConn)
			print "interConn is " + str(interConn)
			interCoeff = interCoeff + interConn/avgConn
		else:
			print avgConn
			interCoeff = interCoeff + avgConn
		print "==========================="
print interCoeff
spi = intraCoeff/interCoeff
s = '30' + ',' + '30' + ',' + str(spi) + '\n'
print "spi is " + str(spi)
file = open('/home/hao/newworkspaces/StateFinder-master/result_30/spi_result.csv', 'a')
file.write(s)
file.close()
