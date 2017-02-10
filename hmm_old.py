# -*- coding: utf-8 -*-
import math
import pdb
from itertools import izip
from math import log
from scipy import stats
from scipy.stats import norm
from scipy import integrate
import numpy as np
from calInit import DataInit
from update_hmm import UpdateHmm
data = DataInit('/home/hao/py-work/PythonHMM/test_new_int_raw.csv', (), (), [], '/home/hao/py-work/PythonHMM/test_new_rle.csv', '/home/hao/py-work/PythonHMM/test_new_symbol.csv')
#data = UpdateHmm('./test_resultraw2.log')
def norm_function(x, mu, std):
    return 1/(std*(2*np.pi)**0.5)*np.e**(-(x-mu)**2/(2*std**2))

def function_kde(x, distrubList, n, h):
    func = 0
    for i in range(0, n):
        func = func + 1/(n*h*(2*np.pi)**0.5)*np.e**(-(x-distrubList[i])**2/(2*h**2))
    return func

# def best_pattern(beginTime, endTime):
#     #begin ~ end all of point value
    
#     return pattern



def write_result_part2(current_d, current_t, current_oldState, prevd, prevs, prePattern):
    length = current_t-current_d
    maxState = current_oldState
    t = length
    d = 0
    oldState = 0
    pattern = 0.0
    i = len(prevd) - 1
    j = len(prevs) - 1
    k = len(prePattern) - 1
    while i >= 0:
        if (prevd[i][2] == t) and (prevd[i][1] == maxState):
            d = prevd[i][0]
            break
        i -= 1
    while j >= 0:
        if (prevs[j][3] == t) and (prevs[j][2] == maxState) and (prevs[j][0] == d):
            oldState = prevs[j][1]
            break
        j -= 1
    while k >= 0:
        if((prePattern[k][2] == t) and (prePattern[k][0] == d)):
            pattern = prePattern[k][1]
            break
        k -= 1
    resultList = [d , t , maxState , oldState , pattern]    
    s = ''
    n = 0
    while n < len(resultList):
        s = s + str(resultList[n]) + str(',')
        n += 1
    s = s.rstrip(',')
    s = s + str('\n')
    file = open('./test_dataraw3.log', 'a')
    file.write(s)
    file.close()
    if t-d == 0:
        print "finish!!!!"
    else:
        write_result_part2(d, t, oldState, prevd, prevs, prePattern)


def _normalize_prob(prob, item_set):
    result = {}
    if prob is None:
        number = len(item_set)
        for item in item_set:
            result[item] = 1.0 / number
    else:
        prob_sum = 0.0
        for item in item_set:
            prob_sum += prob.get(item, 0)

        if prob_sum > 0:
            for item in item_set:
                result[item] = prob.get(item, 0) / prob_sum
        else:
            for item in item_set:
                result[item] = 0

    return result

def _normalize_prob_two_dim(prob, item_set1, item_set2):
    result = {}
    if prob is None:
        for item in item_set1:
            result[item] = _normalize_prob(None, item_set2)
    else:
        for item in item_set1:
            result[item] = _normalize_prob(prob.get(item), item_set2)

    return result

def _count(item, count):
    if item not in count:
        count[item] = 0
    count[item] += 1

def _count_two_dim(item1, item2, count):
    if item1 not in count:
        count[item1] = {}
    _count(item2, count[item1])

def _get_init_model(sequences):
    symbol_count = {}
    state_count = {}
    state_symbol_count = {}
    state_start_count = {}
    state_trans_count = {}

    for state_list, symbol_list in sequences:
        pre_state = None
        for state, symbol in izip(state_list, symbol_list):
            _count(state, state_count)
            _count(symbol, symbol_count)
            _count_two_dim(state, symbol, state_symbol_count)
            if pre_state is None:
                _count(state, state_start_count)
            else:
                _count_two_dim(pre_state, state, state_trans_count)
            pre_state = state

    return Model(state_count.keys(), symbol_count.keys(),
        state_start_count, state_trans_count, state_symbol_count)

def train(sequences, delta=0.0001, smoothing=0):
    """
    Use the given sequences to train a HMM model.
    This method is an implementation of the `EM algorithm
    <http://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm>`_.

    The `delta` argument (which is defaults to 0.0001) specifies that the
    learning algorithm will stop when the difference of the log-likelihood
    between two consecutive iterations is less than delta.

    The `smoothing` argument is used to avoid zero probability,
    see :py:meth:`~hmm.Model.learn`.
    """

    model = _get_init_model(sequences)
    length = len(sequences)

    old_likelihood = 0
    for _, symbol_list in sequences:
        old_likelihood += log(model.evaluate(symbol_list))

    old_likelihood /= length

    while True:
        new_likelihood = 0
        for _, symbol_list in sequences:
            model.learn(symbol_list, smoothing)
            new_likelihood += log(model.evaluate(symbol_list))

        new_likelihood /= length

        if abs(new_likelihood - old_likelihood) < delta:
            break

        old_likelihood = new_likelihood

    return model

class Model(object):
    """
    This class is an implementation of the Hidden Markov Model.

    The instance of this class can be created by passing the given states,
    symbols and optional probability matrices.

    If any of the probability matrices are not given, the missing matrics
    will be set to the initial uniform probability.
    """

    def __init__(self, states, symbols, rawList, start_prob, trans_prob, segma):
        self._states = states
        self._symbols = symbols
        self.rawList = rawList
        self._start_prob = start_prob
        self._trans_prob = trans_prob
        self.segma = segma

    def __repr__(self):
        return '{name}({_states}, {_symbols}, {_start_prob}, {_trans_prob}, {_emit_prob})' \
            .format(name=self.__class__.__name__, **self.__dict__)

    def states(self):
        """ Return the state set of this model. """
        return set(self._states)

    def states_number(self):
        """ Return the number of states. """
        return len(self._states)

    def symbols(self):
        """ Return the symbol set of this model. """
        return set(self._symbols)

    def symbols_number(self):
        """ Return the number of symbols. """
        return len(self._symbols)

    def start_prob(self, state):
        """
        Return the start probability of the given state.

        If `state` is not contained in the state set of this model, 0 is returned.
        """
        if state not in self._states:
            return 0
        return self._start_prob[state]

    def trans_prob(self, state_from, state_to):
        """
        Return the probability that transition from state `state_from` to
        state `state_to`.

        If either the `state_from` or the `state_to` are not contained in the
        state set of this model, 0 is returned.
        """
        if state_from not in self._states or state_to not in self._states:
            return 0
        return self._trans_prob[state_from][state_to]
    
    def bestPattern(self, beginTime, endTime):
        # 这里的beginTime和endTime是指RLE序列后的pattern的序列,并不是从原始序列出发,寻找pattern的过程
        #valuesArr = data.getMeansBySymbol()
        rawValuesArr = self.rawList
        pattern = 0
        #get patttern's value by estimate
        pattern = sum(rawValuesArr[(beginTime):(endTime)])/float((endTime-beginTime))
        return pattern
    
    # def relativeErrPattern(self, beginTime, endTime, pattern, rawFile, rleFile):
    #     rawValueArr = []
    #     relativeErrorSum = 0.0
    #     newRleFile = open(rleFile)
    #     for rleLine in newRleFile.readlines()[beginTime:endTime]:
    #         rleArr = rleLine.strip('\n').split(',')
    #         newRawFile = open(rawFile)
    #         for rawLine in newRawFile.readlines()[(int(rleArr[0])/10):(int(rleArr[1])/10)]:
    #             rawArr = rawLine.strip('\n').split(',')
    #             rawValueArr.append(int(rawArr[2]))
    #     length  = len(rawValueArr)
    #     for i in xrange(0, length):
    #         relativeErrorSum = relativeErrorSum + math.fabs((pattern-float(rawValueArr[i]))/float(rawValueArr[i]))
    #     relativeError = relativeErrorSum/length
    #     print "relativeError is", realtiveError
    #     return relativeError

    def relativeErr(self, beginTime, endTime, pattern):
        relativeErrorSum = 0.0
        rawValuesArr = self.rawList
        patternValuesArr = rawValuesArr[(beginTime):(endTime)] #计算此步pattern中所含的原始序列值 
        length = len(patternValuesArr)
        for i in xrange(0, length):
            # if math.fabs(pattern - float(patternValuesArr[i])/float(patternValuesArr[i])) > self.segma:
            if math.fabs((pattern - float(patternValuesArr[i]))/pattern) > 1.5:
                relativeError = 10
                break
            elif (pattern/float(patternValuesArr[i]) < 0.05) or (float(patternValuesArr[i])/pattern < 0.05):
                relativeError = 10
                break             
            else:
                relativeErrorSum = relativeErrorSum + math.fabs((pattern-float(patternValuesArr[i]))/pattern)
                relativeError = relativeErrorSum/length
        #print "relativeError is", relativeError
        return relativeError

    # def relativeErr(self, beginTime, endTime, pattern):
    #     relativeError = 0.0
    #     rawValuesArr = self.rawList
    #     patternValuesArr = rawValuesArr[(beginTime):(endTime)] #计算此步pattern中所含的原始序列值 
    #     length = len(patternValuesArr)
    #     for i in xrange(0, length):
    #         if math.fabs(pattern - float(patternValuesArr[i])/float(patternValuesArr[i]))/float(patternValuesArr[i]) > self.segma:
    #             relativeError = 100
    #             break
    #         else:
    #             relativeError = 0.0
    #     return relativeError     
        #print "relativeError is", relativeError


    def emit_prob(self, state, beginTime, endTime, pattern):
        #print "emit_prob=================="
        # if state not in self._states or symbol not in self._symbols:
        #     return 0

        # first turn
        #varDict = data.getVarianceByCluster()

        patterns = data.findPattern()
        varDict = data.getVarianceByCluster(patterns)
        #distrubList, h, n = data.calDistrubitionByMeans(state)
        stateMeans = varDict[state][0]
        std = varDict[state][1]
        #state pattern
        stateMaxPoint = varDict[state][2]
        stateMinPoint = varDict[state][3]
        # distrubMaxPoint = max(distrubList)
        # distrubMinPoint = min(distrubList)
        #interval = sum(valuesArr[(indexBegin):(indexEnd)])/(indexEnd-indexBegin)
        interval = pattern
        #distrubInterval = math.fabs(stateMeans - interval)
        plusValue = (stateMaxPoint - stateMinPoint)/10
        #interval = valuesArr[index]
        # 不在state范围内不进行计算
        if (interval > stateMaxPoint) or (interval < stateMinPoint):
          # print "outProb is", 0
          # print "emit_prob================== not in "
          return 0
        else:
          bInter = 0
          eInter = 0
          outProb = 1
          err = 0
          # get interval
          # to be fixed
          for i in range(10):
            bInter = stateMinPoint + i*plusValue
            eInter = stateMinPoint + (i+1)*plusValue
            if (interval > bInter and interval <= eInter):
                break
            elif (interval < bInter or interval > eInter):
                bInter = 0
                eInter = 0
          outProb , err = integrate.quad(norm_function, bInter, eInter, (stateMeans, std))
          # print "outProb is:",outProb
          # print "emit_prob================== in"
          return outProb









    def _forward(self, sequence):
        sequence_length = len(sequence)
        if sequence_length == 0:
            return []

        alpha = [{}]
        for state in self._states:
            alpha[0][state] = self.start_prob(state) * self.emit_prob(state, sequence[0])

        for index in xrange(1, sequence_length):
            alpha.append({})
            for state_to in self._states:
                prob = 0
                for state_from in self._states:
                    prob += alpha[index - 1][state_from] * \
                        self.trans_prob(state_from, state_to)
                alpha[index][state_to] = prob * self.emit_prob(state_to, sequence[index])

        return alpha

    def _backward(self, sequence):
        sequence_length = len(sequence)
        if sequence_length == 0:
            return []

        beta = [{}]
        for state in self._states:
            beta[0][state] = 1

        for index in xrange(sequence_length - 1, 0, -1):
            beta.insert(0, {})
            for state_from in self._states:
                prob = 0
                for state_to in self._states:
                    prob += beta[1][state_to] * \
                        self.trans_prob(state_from, state_to) * \
                        self.emit_prob(state_to, sequence[index])
                beta[0][state_from] = prob

        return beta

    def evaluate(self, sequence):
        """
        Use the `forward algorithm
        <http://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm>`_
        to evaluate the given sequence.
        """
        length = len(sequence)
        if length == 0:
            return 0

        prob = 0
        alpha = self._forward(sequence)
        for state in alpha[length - 1]:
            prob += alpha[length - 1][state]

        return prob

    def decode(self, sequence):
        """
        Decode the given sequence.

        This method is an implementation of the
        `Viterbi algorithm <http://en.wikipedia.org/wiki/Viterbi_algorithm>`_.
        """
        sequence_length = len(sequence)
        if sequence_length == 0:
            return []

        delta = {}
        for state in self._states:
            delta[state] = self.start_prob(state) * self.emit_prob(state, sequence[0])

        pre = []
        for index in xrange(1, sequence_length):
            delta_bar = {}
            pre_state = {}
            for state_to in self._states:
                max_prob = 0
                max_state = None
                for state_from in self._states:
                    prob = delta[state_from] * self.trans_prob(state_from, state_to)
                    if prob > max_prob:
                        max_prob = prob
                        max_state = state_from
                delta_bar[state_to] = max_prob * self.emit_prob(state_to, sequence[index])
                pre_state[state_to] = max_state
            delta = delta_bar
            pre.append(pre_state)

        max_state = None
        max_prob = 0
        for state in self._states:
            if delta[state] > max_prob:
                max_prob = delta[state]
                max_state = state

        if max_state is None:
            return []

        result = [max_state]
        for index in xrange(sequence_length - 1, 0, -1):
            max_state = pre[index - 1][max_state]
            result.insert(0, max_state)

        return result

    #numOfTime wenben changdu
    def ex_decoding(self, sequence, rawFile, rleFile):
        numOfTime = len(sequence)
        if numOfTime == 0:
            return []
        # t's range may have some difference
        delta = {} #output prob
        prePattern = []
        prevd = []
        prevs = [] #result 
        newTemp = 0
        currentTemp = 0
        newStateTemp = 0
        currentStateTemp = 0
        stateResult = 0
        temp = 0 #interval variable
        for t in xrange(1, numOfTime+1):
            for state in self._states:
                delta['delta' + str(t) + '(' + str(state) + ')'] = 0
                for d in xrange(1, t+1):
                    #print "wwwwwwwwwwwwwwwwwwwwwww start"
                   # print "d is", d,", t is", t, ", state is", state
                    pattern = self.bestPattern(t-d,t)
                    if self.relativeErr(t-d, t, pattern) > self.segma:
                        print "d is",d, ",t is",t, ",state is", state
                        print "break error---"
                        break
                    else:
                        # we got pattern value and time range
                        prePattern.append([d,pattern,t])
                        if d == t:
                            temp = self.start_prob(state) * self.emit_prob(state, t-d, t, pattern)# how to cal the prob??
                        else:
                            for oldState in self._states:
                                # print "d < t =============================="
                                # print "oldState is",oldState
                                # print "current oldState delta is", delta['delta' + str(t-d) + '(' + str(oldState) + ')']
                                newTemp = delta['delta' + str(t-d) + '(' + str(oldState) + ')'] * self.trans_prob(oldState , state) * self.emit_prob(state, t-d, t, pattern)
                                #print "newTemp is", newTemp
                                if newTemp > currentTemp:
                                    currentTemp = newTemp
                                    prevs.append([d,oldState,state,t])
                                    print currentTemp
                                    print [d,oldState,state,t]
                                #print "d < t =============================="
                            temp = currentTemp
                        #print "temp current value:",temp
                        # s1 = str(t) + str(',') + str(state) + str(',') + str(temp) + str(',') + str(delta['delta' + str(t) + '(' + str(state) + ')']) + str('\n')
                        # fileTemp = open('/home/hao/py-work/PythonHMM/hmm_old_logs/hmm_old_final_temp_vs_delta_t.log','a')
                        # fileTemp.write(s1)
                        # fileTemp.close()
                        currentTemp = 0
                        if temp > delta['delta' + str(t) + '(' + str(state) + ')']:
                            print "                                             "
                            print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
                            print "we got d and t is", [d,t], ", state is", state, ", pattern is", pattern
                            print temp
                            print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
                            delta['delta' + str(t) + '(' + str(state) + ')'] = temp
                            prevd.append([d,state,t,pattern])
                            print "                        "
        #print delta['delta' + str(96) + '(' + str(0) + ')'], delta['delta' + str(40) + '(' + str(1) + ')'], delta['delta' + str(40) + '(' + str(2) + ')'], delta['delta' + str(40) + '(' + str(3) + ')'], delta['delta' + str(40) + '(' + str(4) + ')']#, delta['delta' + str(96) + '(' + str(5) + ')'],delta['delta' + str(96) + '(' + str(6) + ')']
        return prevd, prevs, prePattern, delta


    def write_result(self, prevd, prevs, prePattern, delta, length):
        states = self._states
        maxProb = 0.0
        maxState = 0
        oldState = 0
        pattern = 0.0
        # judge the max outProb
        for state in states:
            if delta['delta' + str(length) + '(' + str(state) + ')'] > maxProb:
                maxProb = delta['delta' + str(length) + '(' + str(state) + ')']
                maxState = state
        t = length
        d = 0
        i = len(prevd) - 1
        j = len(prevs) - 1
        k = len(prePattern) - 1
        while i > 0:
            if (prevd[i][2] == t) and (prevd[i][1] == maxState):
                d = prevd[i][0]
                break
            i -= 1
        while j > 0:
            if (prevs[j][3] == t) and (prevs[j][2] == maxState) and (prevs[j][0] == d):
                oldState = prevs[j][1]
                break
            j -= 1
        while k > 0:
            if((prePattern[k][2] == t) and (prePattern[k][0] == d)):
                pattern = prePattern[k][1]
                break
            k -= 1
        resultList = [d , t , maxState , oldState , pattern]    
        s = ''
        n = 0
        while n < len(resultList):
            s = s + str(resultList[n]) + str(',')
            n += 1
        s = s.rstrip(',')
        s = s + str('\n')
        file = open('./test_dataraw3.log', 'a')
        file.write(s)
        file.close()
        return d, t, oldState

    def learn(self, sequence, smoothing=0):
        """
        Use the given `sequence` to find the best state transition and
        emission probabilities.

        The optional `smoothing` argument (which is defaults to 0) is the
        smoothing parameter of the
        `additive smoothing <http://en.wikipedia.org/wiki/Additive_smoothing>`_
        to avoid zero probability.
        """
        length = len(sequence)
        alpha = self._forward(sequence)
        beta = self._backward(sequence)

        gamma = []
        for index in xrange(length):
            prob_sum = 0
            gamma.append({})
            for state in self._states:
                prob = alpha[index][state] * beta[index][state]
                gamma[index][state] = prob
                prob_sum += prob

            if prob_sum == 0:
                continue

            for state in self._states:
                gamma[index][state] /= prob_sum

        xi = []
        for index in xrange(length - 1):
            prob_sum = 0
            xi.append({})
            for state_from in self._states:
                xi[index][state_from] = {}
                for state_to in self._states:
                    prob = alpha[index][state_from] * beta[index + 1][state_to] * \
                        self.trans_prob(state_from, state_to) * \
                        self.emit_prob(state_to, sequence[index + 1])
                    xi[index][state_from][state_to] = prob
                    prob_sum += prob

            if prob_sum == 0:
                continue

            for state_from in self._states:
                for state_to in self._states:
                    xi[index][state_from][state_to] /= prob_sum

        states_number = len(self._states)
        symbols_number = len(self._symbols)
        for state in self._states:
            # update start probability
            self._start_prob[state] = \
                (smoothing + gamma[0][state]) / (1 + states_number * smoothing)

            # update transition probability
            gamma_sum = 0
            for index in xrange(length - 1):
                gamma_sum += gamma[index][state]

            if gamma_sum > 0:
                denominator = gamma_sum + states_number * smoothing
                for state_to in self._states:
                    xi_sum = 0
                    for index in xrange(length - 1):
                        xi_sum += xi[index][state][state_to]
                    self._trans_prob[state][state_to] = (smoothing + xi_sum) / denominator
            else:
                for state_to in self._states:
                    self._trans_prob[state][state_to] = 0

            # update emission probability
            gamma_sum += gamma[length - 1][state]
            emit_gamma_sum = {}
            for symbol in self._symbols:
                emit_gamma_sum[symbol] = 0

            for index in xrange(length):
                emit_gamma_sum[sequence[index]] += gamma[index][state]

            if gamma_sum > 0:
                denominator = gamma_sum + symbols_number * smoothing
                for symbol in self._symbols:
                    self._emit_prob[state][symbol] = \
                        (smoothing + emit_gamma_sum[symbol]) / denominator
            else:
                for symbol in self._symbols:
                    self._emit_prob[state][symbol] = 0






# a1 = data.stateFinder()
# b1 = data.symbolFinder()
# c1 = data.calStartProb()
# d1 = data.transProb()
# e1 = data.listFinder()
# var = data.getVarianceByCluster()
# val = data.getMeansBySymbol()
a1 = tuple(data.stateList)
b1 = tuple(data.stateList)
c1 = data.calStartProb()
d1 = data.calTransProb()
e1 = data.findAllPattern()
print a1
print b1
print c1
print d1
print e1
# print var
# print val

ss = Model(a1, b1, e1, c1, d1, 0.7)
prevd, prevs, prePattern, delta = ss.ex_decoding(e1, '/home/hao/py-work/PythonHMM/test_new_int_raw2.csv', '/home/hao/py-work/PythonHMM/test_new_rle2.csv')
print prevd
print prevs
print prePattern
d, t, oldState = ss.write_result(prevd, prevs, prePattern, delta, 19)
write_result_part2(d,t,oldState,prevd,prevs,prePattern)


