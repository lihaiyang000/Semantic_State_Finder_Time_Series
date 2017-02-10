# -*- coding: utf-8 -*-

from hmm import Model
from numpy import *
import csv
fileName = '/home/hao/newworkspaces/StateFinder-master/data-rle.csv'
listTest = []
symbolTest = []
stateTest = []
with open(fileName) as f:
	reader = csv.reader(f)
	for row in reader:
		listTest.append(row[2])
# load file

symbolTest = unique(listTest)

for index in range(len(symbolTest)):
	stateTest.append('a' + symbolTest[index])

states = tuple(stateTest) #change into init states	
symbols = tuple(symbolTest) #change into patterns
print states
print symbols
# states = ('rainy' , 'sunny')
# symbols = ('walk' , 'shop' , 'clean')

startTestProb = {}

for index in range(len(states)):
	prob = 1.0/len(states)
	startTestProb[states[index]] = prob
# init pi start_probaility
start_prob = dict(sorted(startTestProb.items(), key=lambda e:e[0], reverse=False))#sorted return a list not a dictionary and use dict we got a dictionary
print start_prob

# start_prob = {
# 	'rainy': 0.5,
# 	'sunny': 0.5
# }

trans_prob = {
    'rainy': { 'rainy' : 0.5, 'sunny' : 0.5 },
    'sunny': { 'rainy' : 0.3, 'sunny' : 0.7 }
} # # #

product_prob = {
    'rainy': { 'walk' : 0.1, 'shop' : 0.4, 'clean' : 0.5 },
    'sunny': { 'walk' : 0.6, 'shop' : 0.3, 'clean' : 0.1 }
} # normal disturbition

sequence = listTest
print sequence
# sequence = ['walk' , 'shop' , 'shop' , 'clean' , 'walk']
# model = Model(states, symbols, start_prob, trans_prob, product_prob)

# print model.evaluate(sequence)
# print model.decode(sequence)

