#!/usr/bin/env python
# coding: utf-8

import PyRANTAPI as RANT

Softp = RANT.NNm (5, 16, 14) # 5 layers, 16 inputs and 14 outputs

process = [True] * 36

O = Softp.LoadFile ("/Users/dsantry/Scratch/Data/water-quality-nom.csv", 16, 14,process)
# Softp.DisplayData (O)

Ndata = Softp.DataLen (O)   # Number of rows in the data set

Softp.AddPreProcessing (O)
Softp.AddDense (75, True)   # 75 nodes, True connotes use ADAM
Softp.AddDense (75, True)
Softp.AddMultiCLayer ()

Softp.SetSGD (0.03)         # Use 3% of the training set per SGD epoch
Softp.SetStopLoss (0.0025)

Softp.SetKeepAlive (10000)  # Print the current loss every 100000 epochs
Softp.Arch ()               # Print the ANN's architecture

Softp.Train (O, 189468)

wrong = 0

for i in range (0, Ndata):

	y = Softp.AnswerVec (O, i)

	# This is multilabel, so no single prediction
	guess = Softp.ClassifyVec (Softp.ExtractTuple (O, i))

	# Compare the ground truth with inference per label
	for j in range (14):
		z = round (guess[j])
		if z != y[j]:
			wrong += 1
			break

print ("Incorrect\t" + str (wrong))
print ("Loss\t\t" + str (Softp.Loss ()))
print ("Steps\t\t" + str (Softp.Steps ()))

