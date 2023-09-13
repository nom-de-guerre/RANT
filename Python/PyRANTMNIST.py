#!/usr/bin/env python
# coding: utf-8

# import sys 
# import os
# sys.path.append(os.path.abspath("/Users/hp/Documents/AFR"))


import time

import PyRANTAPI as RANT

Softp = RANT.NNm (7, 784, 10)

O = Softp.LoadMNIST ("../../../Neural Networks/Data/MNIST/")

Softp.AddPreProcessing (O)
Softp.AddConvFilter (5, 3, 1)
Softp.AddConvMaxPool (5, 2, 1)
Softp.AddDense (50, True)
Softp.AddDense (50, True)
Softp.AddSoftmax (True)


# O = Softp.TestSet ()

Softp.SetSGD (0.005)
Softp.SetStopLoss (0.05)
Softp.SetKeepAlive (100)

Softp.Arch ()
Softp.Shape ()

start = time.time ()

Softp.Train (O, 5000)

train_dt = time.time () - start
train_dt /= Softp.Steps ()

start = time.time ()

wrong = 0

for i in range (0,150):

    y = Softp.Answer (O, i)
    guess = Softp.Classify (Softp.ExtractTuple (O, i))
    if y != guess:
        print (i)
        wrong += 1

inference_dt = time.time () - start

print ("Incorrect\t" + str (wrong))
print ("Loss\t\t" + str (Softp.Loss ()))
print ("Steps\t\t" + str (Softp.Steps ()))
print ("Train dt\t" + str (train_dt))
print ("Test dt\t\t" + str (inference_dt))

