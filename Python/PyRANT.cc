/*

Copyright (c) 2022, Douglas Santry
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, is permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <sys/param.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include <NNm.h>
#include <options.h>
#include <read_csv.h>
#include <MNIST.h>

#define N_POINTS	32

#define PI          3.141592653589793
#define PI_2        1.570796326794897

#define RANGE       PI_2

extern "C"
{

void SeedRandom (long seed)
{
	srand (seed);
}

NNet_t * AllocNNm (int Nhidden, int Nin, int Nout)
{
	NNet_t *Np = NULL;

	Np = new NNet_t (Nhidden, Nin, Nout);

	Np->SetKeepAlive (0);

	return Np;
}

void SetKeepAlive (NNet_t *Np, int modulus)
{
	Np->SetKeepAlive (modulus);
}

void SetStopLoss (NNet_t *Np, double halt)
{
	Np->SetHalt (halt);
}

void SetMaxIterations (NNet_t *Np, int N)
{
	Np->SetMaxIterations (N);
}

void SetUseSGD (NNet_t *Np, double percentBatchSize)
{
	Np->SetSGD (percentBatchSize);
}

void AddDense (NNet_t *Np, int N, bool useADAM)
{
	Np->AddDenseLayer (N, (useADAM ? ADAM : RPROP));
}

void AddMSE (NNet_t *Np, bool useADAM)
{
	Np->AddScalerMSELayer (useADAM ? ADAM : RPROP);
}

void AddFilter (NNet_t *Np, int N, int width, int stride, bool useADAM)
{
	Np->Add2DFilterLayer (N, width, stride, useADAM ? ADAM : RPROP);
}

void AddMaxPool (NNet_t *Np, int N, int width, int stride)
{
	Np->Add2DMaxPoolLayer (N, width, stride);
}

void AddSoftmax (NNet_t *Np, bool useADAM)
{
	Np->AddSoftmaxLayer (useADAM ? ADAM : RPROP);
}

void AddMultiCLayer (NNet_t *Np, bool useADAM)
{
	Np->AddMultiCLayer (useADAM ? ADAM : RPROP);
}

void Shape (NNet_t *Np)
{
	Np->DisplayShape ();
}

void Arch (NNet_t *Np)
{
	Np->DisplayModel ();
}

void Train (NNet_t *Np, DataSet_t *O, int steps)
{
	try {

		Np->Train (O, steps);

	} catch (const char *msgp) {

		printf ("%s\n", msgp);
	}
}

double Inference (NNet_t *Np, double *predictors)
{
	return Np->Compute (predictors);
}

int Classify (NNet_t *Np, double *predictors)
{
	return (int) Np->Compute (predictors);
}

double * ClassifyVec (NNet_t *Np, double *predictors)
{
	double *y = Np->ComputeVec (predictors);

	return y;
}

void SetPreProcessing (NNet_t *Np, DataSet_t *O)
{
	Np->AddPreProcessingLayer (O);
}

double Loss (NNet_t *Np)
{
	return Np->Loss ();
}

int Steps (NNet_t *Np)
{
	return Np->Steps ();
}

DataSet_t *LoadMNIST (char const * const path)
{
	char fullpath_data [MAXPATHLEN];
	char fullpath_labels [MAXPATHLEN];

	snprintf (fullpath_data, MAXPATHLEN, "%s/train-images.idx3-ubyte", path);
	snprintf (fullpath_labels, MAXPATHLEN, "%s/train-labels.idx1-ubyte", path);
    MNIST_t data (fullpath_data, fullpath_labels);

	return data.Data ();
}

DataSet_t *LoadCSVClass (char const * const fileName, 
	const int N, 
	int *accept)
{
	LoadCSV_t Z (fileName);
	bool *columns = new bool [N];

	for (int i = 0; i < N; ++i)
		columns[i] = accept[i] ? true : false;

	DataSet_t *O = Z.LoadDS (N, columns);

	delete [] columns;

	return O;
}


DataSet_t *LoadCSV (char const * const fileName, 
	const int N, 
	const int Nin,
	const int Nout,
	int *accept)
{
	LoadCSV_t Z (fileName);
	bool *columns = new bool [N];

	for (int i = 0; i < N; ++i)
		columns[i] = accept[i] ? true : false;

	DataSet_t *O = Z.LoadDS (N, columns);
	O->Split (Nin, Nout); // let data know length of answer

	delete [] columns;

	return O;
}

void DisplayData (DataSet_t const * const O)
{
	O->Display ();
}

DataSet_t *BuildTrainingSet (int N)
{
	DataSet_t *O = new DataSet_t (N, 1, 1);

	for (int i = 0; i < N; ++i)
	{
		double sample = (double) rand () / RAND_MAX;

		(*O)[i][0] = sample;
		(*O)[i][1] = sin ((*O)[i][0] * RANGE);
	}

	return O;
}

double *ExtractTuple (DataSet_t *O, int index)
{
	return (*O)[index];
}

double Answer (DataSet_t *O, int index)
{
	return O->Answer (index);
}

double *AnswerVec (DataSet_t *O, int index)
{
	double *p = O->AnswerVec (index);

	return p;
}

int DataLen (DataSet_t *O)
{
	return O->N ();
}

}

