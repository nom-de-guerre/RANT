#!/usr/bin/env python
# coding: utf-8

import PyRANTAPI as RANT

Softp = RANT.NNm (7, 784, 10)  # 7 layers, 784 inputs, 10 outputs (the 10 possible digits)

O = Softp.LoadMNIST ("../../../Neural Networks/Data/MNIST/")

Softp.AddPreProcessing (O)
Softp.AddConvFilter (5, 3, 1)      # 5 filters, width of 3, stride 1
Softp.AddConvMaxPool (5, 2, 1)     # 5 input maps, width of 2, stride 1
Softp.AddDense (50, True)          # 50 perceptrons, True directs the use of ADAM
Softp.AddDense (50, True)
Softp.AddSoftmax (True)

Softp.SetSGD (0.005)               # Use 0.5% of the training set per SGD training epoch
Softp.SetStopLoss (0.05)
Softp.SetKeepAlive (100)

Softp.Arch ()
Softp.Shape ()

Softp.Train (O, 5000)

wrong = 0

for i in range (0,5000):

    y = int (Softp.Answer (O, i))
    guess = Softp.Classify (Softp.ExtractTuple (O, i))
	# Compare ground truth with inference, print the mistakes
    if y != guess:
        print (i, y, guess)
        wrong += 1

print ("Incorrect\t" + str (wrong))
print ("Loss\t\t" + str (Softp.Loss ()))
print ("Steps\t\t" + str (Softp.Steps ()))

