#!/usr/bin/env python
# coding: utf-8

# import sys 
# import os
# sys.path.append(os.path.abspath("/Users/hp/Documents/AFR"))


import time
from statistics import mean as mean
from statistics import stdev as stdev

import PyRANTAPI as RANT

Softp = RANT.NNm (5, 16, 14)

process = [ True, True, True, True, True, True,
True, True, True, True, True, True,
True, True, True, True, True, True,
True, True, True, True, True, True,
True, True, True, True, True, True,
True, True, True, True, True, True]

O = Softp.LoadFile ("/Users/dsantry/Scratch/Data/water-quality-nom.csv", 16, 14,process)
# Softp.DisplayData (O)

Ndata = Softp.DataLen (O)

Softp.AddPreProcessing (O)
Softp.AddDense (75, True)
Softp.AddDense (75, True)
Softp.AddMultiCLayer ()

# O = Softp.TestSet ()

Softp.SetSGD (0.03)
Softp.SetStopLoss (0.0025)

Softp.SetKeepAlive (10000)
Softp.Arch ()

start = time.time ()

# Softp.Train (O, 10)
Softp.Train (O, 189468)

train_dt = time.time () - start
train_dt /= Softp.Steps ()

start = time.time ()

wrong = 0
micro = []

for i in range (0, Ndata):

	guess_dt = time.time ()
	y = Softp.AnswerVec (O, i)
	guess = Softp.ClassifyVec (Softp.ExtractTuple (O, i))

	for j in range (14):
		z = round (guess[j])
		if z != y[j]:
			wrong += 1
			break

	guess_dt = time.time () - guess_dt
	micro.append (guess_dt *1000000)

inference_dt = time.time () - start

print ("Incorrect\t" + str (wrong))
print ("Loss\t\t" + str (Softp.Loss ()))
print ("Steps\t\t" + str (Softp.Steps ()))
print ("Train dt\t" + str (train_dt))
print ("Test dt\t\t" + str (inference_dt))

print (mean (micro)*1000000, stdev (micro)*1000000) # µsec

print (micro)

