# PythonHMM

**PythonHMM** is a python implementation of the semantic finding by using HMM model

## Usage

1. clustering
    cd Spclust
    run: java -jar SpComputeSymbols -h for more details
    we got the clustering result

2. refinement by HMM model

    use the result in clustering, change the parameters in hmm_old.py --12 line
    data = DataInit('/home/hao/py-work/PythonHMM/test_new_int_raw.csv', (), (), [], '/home/hao/py-work/PythonHMM/test_new_rle.csv', '/home/hao/py-work/PythonHMM/test_new_symbol.csv')

    run: python hmm_old.py and get the result

3. run: python update_hmm.py to update the parameters in HMM model

4. refinement HMM model
    change the parameters in hmm_old.py -- 12 line
    data = UpdateHmm('./test_resultraw2.log')

    run: python hmm_old.py and get the result

If the lastest result is the same as the last one, wo got the final semantic result 

运行spclust后，得到所需数据文件。
第一步:运行hmm_for_cmapss.py 获得init步骤后的结果
其中需要更改的参数有:
16行,需要更改其中地址参数
data = DataInit('原始数据序列文件绝对地址', (), (), [], '通过Spclust后的长度压缩编码序列文件绝对地址', '通过Spclust后的symbol序列文件绝对地址')

69行,更改输出文件的地址file = open('输出文件地址\输出文件名', 'a')。输出文件可以不用提前创建，但是要注意输出文件地址后面一定要有文件名，这样输出文件才能被创建。

577行,需要更改输出文件的地址(绝对地址)，输出地址通69行一致。

700行,需要更改其中地址参数
prevd, prevs, prePattern, delta = ss.ex_decoding(e1, '原始数据序列文件绝对地址', '通过Spclust后的symbol序列文件绝对地址')9
705行,更改最后一个参数,指的是原始序列的长度
d, t, oldState = ss.write_result(prevd, prevs, prePattern, delta, 原始序列的长度)


第二步:运行result_sequence.py,获得相关的序列
2行,更改为第一步中输出的结果文件的绝对地址,并删除结果文件中第一行数据（该数据为hmm模型的最佳概率）以方便后续步骤使用。

8行,修改地址	
rawFile = open('原始数据序列的文件的绝对地址')

15行，修改输出文件地址。此处需要提前建立输出文件,否则无法输出文件
	with open('输出文件的绝对地址', 'r+') as f:


第三步:运行float_to_int.py,获得整数型的序列
2行,更改为第二步中的输出的结果文件的绝对地址

10行，修改输出地址，通过viterbi算法后输出的观测序列文件的绝对地址，并且创建文件名。

13行,修改输出地址，通过viterbi算法后输出的symbol序列文件的绝对地址，并且创建文件名。

第四步:根据hmm_update.py,得到下一步的结果
其中在hmm_old中,有需要更改的参数和位置
测试用的数据只需要运行一次hmm_for_cmapss即可得到结果,不需要用到此步。


需要用到的python的依赖:
scipy,numpy

