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



