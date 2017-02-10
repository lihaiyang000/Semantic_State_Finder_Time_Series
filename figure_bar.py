from calInit import DataInit
import numpy as np
import matplotlib.pyplot as plt

data = DataInit('/home/hao/py-work/PythonHMM/data.csv', (), (), [], '/home/hao/newworkspaces/StateFinder-master/data35-rle.csv', '/home/hao/newworkspaces/StateFinder-master/data35-symbol.csv')
#data = DataInit('/home/hao/newworkspaces/StateFinder-master/StateFinder/input-example/data.csv', (), (), [], '/home/hao/newworkspaces/StateFinder-master/data-rle.csv')

def drawRawData(symbol):
	rawDataDict = data.calRawDataForSymbol()
	plt.hist(rawDataDict[symbol], 50)
	print rawDataDict
	plt.xlabel('Symbol is' + str(symbol))
	plt.ylabel('frequency')
	plt.title('symbol(' +str(symbol) + ') Distribution')
	plt.show()


for i in data.symbolFinder():
	drawRawData(i)



