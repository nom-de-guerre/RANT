#!/usr/bin/env python
# coding: utf-8

import PyRANTAPI as RANT

Softp = RANT.NNm (4, 4, 3)

process = [ True, True, True, True, True ]
O = Softp.LoadFileClass ("../../../Neural Networks/Data/iris.csv", process)

Softp.AddPreProcessing (O) # It needs the training set
Softp.AddDense (20, True)  # 20 nodes, use ADAM, set False for RPROP+
Softp.AddDense (20, True)
Softp.AddSoftmax (True)

Softp.SetStopLoss (0.005)

Softp.Train (O, 7000) # Maximum of 7000 steps to reach the StopLoss

wrong = 0

for i in range (0,150):

    y = Softp.Answer (O, i)
    guess = Softp.Classify (Softp.ExtractTuple (O, i))
    # compare ground truth with inference
    if y != guess:
        wrong += 1

print ("Incorrect\t" + str (wrong))
print ("Loss\t\t" + str (Softp.Loss ()))
print ("Steps\t\t" + str (Softp.Steps ()))

