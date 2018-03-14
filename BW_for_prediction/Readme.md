BW_for_prediction文件夹的使用
1.将state_finder得到的结果带入figure_total_with_bw_result.py代码中
	1.1 首先计算结果分布情况对应的均值和方差,分别写入13行和18行的矩阵中
	1.2 更改20~23行model的数量,与state_finder的类别结果数量一致
	1.3 40行开始以矩阵形式写入state_finder的类别结果
	1.4 80行左右写入当前的测试数据或训练数据
2.得到的结果写入trainX.log和trainY.log
3.test.log中写入对应的测试数据
4.运行LR2.log,使用线性回归的方式获得剩余寿命的预测结果

注: 运行LR2.py需要的依赖:matplotlib.pyplot,numpy,sklearn
