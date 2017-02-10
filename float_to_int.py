resultFloat = open('/home/hao/py-work/PythonHMM/test_new_int_raw2.csv')
resultArr = []
for line in resultFloat.readlines():
	resultArr = line.strip('\n').split(',')
	flo = float(resultArr[2])
	resultInt = int(flo)
	s = resultArr[0] + str(',') + resultArr[1] + str(',') + str(resultInt) + str('\n')
	file = open('./test_new_int_raw3.csv', 'a')
	file.write(s)
	file.close()
resultFloat.close()
