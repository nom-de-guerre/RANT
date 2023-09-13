#!/usr/bin/env python
# coding: utf-8

# import sys 
# import os
# sys.path.append(os.path.abspath("/Users/hp/Documents/AFR"))


import time

import PyRANTAPI as RANT

Softp = RANT.NNm (4, 4, 3)

process = [ False, True, True, True, True, True ]
O = Softp.LoadFileClass ("../../../Neural Networks/Data/iris.csv", process)

Softp.AddPreProcessing (O)
Softp.AddDense (20, True)
Softp.AddDense (20, True)
Softp.AddSoftmax (True)

# O = Softp.TestSet ()

Softp.SetStopLoss (0.0005)

start = time.time ()

Softp.Train (O, 7000)

train_dt = time.time () - start
train_dt /= Softp.Steps ()

start = time.time ()

wrong = 0
micro = []

for i in range (0,150):

    y = Softp.Answer (O, i)
    start_micro = time.time ()
    guess = Softp.Classify (Softp.ExtractTuple (O, i))
    if y != guess:
        wrong += 1
    micro.append ((time.time () - start_micro)*1000000)

inference_dt = time.time () - start

print ("Incorrect\t" + str (wrong))
print ("Loss\t\t" + str (Softp.Loss ()))
print ("Steps\t\t" + str (Softp.Steps ()))
print ("Train dt\t" + str (train_dt))
print ("Test dt\t\t" + str (inference_dt))

print (micro)

