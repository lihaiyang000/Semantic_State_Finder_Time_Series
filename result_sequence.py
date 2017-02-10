# coding: utf-8
hmmResult = open('/home/hao/py-work/PythonHMM/test_dataraw2.log')
hmmArr = []
for hmmLine in hmmResult.readlines():
	hmmArr = hmmLine.strip('\n').split(',')
	endLine = int(hmmArr[1])
	beginLine = int(hmmArr[1]) - int(hmmArr[0])
	rawFile = open('/home/hao/py-work/PythonHMM/test_new_int_raw.csv')
	rawLine = rawFile.readlines()[beginLine:endLine]
	length = len(rawLine)
	rawArrEnd = rawLine[length - 1].strip('\n').split(',')
	rawArrBegin = rawLine[0].strip('\n').split(',')
	s = rawArrBegin[0] + str(',') + rawArrEnd[1] + str(',') + hmmArr[4] + str(',') + hmmArr[2] + str('\n')
	#必须先建立输出文件
	with open('./test_resultraw2.log', 'r+') as f:
		content = f.read()
		f.seek(0,0)
		f.write(s + content)
	rawFile.close()
hmmResult.close()
