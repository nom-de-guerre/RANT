/*

Copyright (c) 2020, Douglas Santry
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

/*
 * The data set for this example can be found here:
 * https://www.kaggle.com/datasets/mathchi/diabetes-data-set
 *
 */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <NNm.h>
#include <read_csv.h>
#include <data.h>
#include <options.h>
#include <confusion.h>

void Run (NNmConfig_t &, int *);
DataSet_t *LoadData (void);

int main (int argc, char *argv[])
{
	NNmConfig_t params;

	int consumed = params.Parse (argc, argv);

	printf ("Seed %ld\n", params.ro_seed);
	srand (params.ro_seed);

	argc -= consumed;
	argv += consumed;

	int N_layers = argc;

	// { length, inputs,  ..., outputs }
	int *layers = new int [N_layers + 1];

	for (int i = 0; i < N_layers; ++i)
		layers[i + 1] = atoi (argv[i]);

	++N_layers; // logits
	layers[0] = N_layers;

	Run (params, layers);

	delete [] layers;
}

void Run (NNmConfig_t &params, int *layers)
{
	DataSet_t *O = LoadData ();
	NNet_t *Np = NULL;
	auto rule = (params.ro_flag ? ADAM : RPROP);
#ifdef __BINARY_CLASS
	int Nout = 1;
#else
	int Nout = 2;
#endif

	// +2, categorical layers and PP
	Np = new NNet_t (layers[0] + 2, 8, Nout);

	Np->AddPreProcessingLayer (O);

	for (int i = 1; i < layers[0]; ++i)
		Np->AddDenseLayer (layers[i], rule);

#ifdef __BINARY_CLASS
	Np->AddScalerMSELayer (rule);
#else
	Np->AddSoftmaxLayer (rule);
#endif

	Np->SetHalt (params.ro_haltCondition);
	Np->SetMaxIterations (params.ro_maxIterations);
	Np->SetAccuracy (); // Halt at 100% accuracy, even if above loss threshold

	Np->DisplayModel ();

	Np->Train (O, params.ro_maxIterations);

	printf ("Loss %f\n", Np->Loss ());

#ifdef __BINARY_CLASS

	int correct = 0;
	int wrong = 0;

	for (int i = 0; i < O->N (); ++i)
	{
		IEEE_t guess = Np->Compute ((*O)[i]);

		printf ("DJS\t%f\t%f\n", guess, O->Answer (i));

		// if (fabs (guess - O->Answer (i)) < 0.5)
		if (guess < 0.5 && O->Answer (i) == 0)
			++correct;
		else if (guess >= 0.5 && O->Answer (i) == 1)
			++correct;
		else {

			++wrong;
//			printf ("WRONG %d\n", i);
		}
	}

	printf ("%d\t%f%%\n",
		correct,
		100.0 * (double) correct / O->N ());

#else

	confusion_t Mo (O, Np);

	Mo.display ();

	printf ("--- TP\tFP\tTN\tFN\n");
	printf ("--- %d\t%d\t%d\t%d\n",
		Mo.GetTP (1),
		Mo.GetFP (1),
		Mo.GetTN (1),
		Mo.GetFN (1));

	double percent = Mo.GetTP (0) + Mo.GetTN (0);
	percent /= (double) O->N ();

	printf ("%d\t%f%%\n",
		Mo.GetTP (0) + Mo.GetTN (0),
		100 * percent);
#endif
}

bool includeFeature [] = {true, true, true, true, true, true, true, true, true};

DataSet_t *LoadData (void)
{
	LoadCSV_t Z ("../Data/diabetes.csv");

	DataSet_t *tp = Z.LoadDS (9, includeFeature);

#ifdef __BINARY_CLASS
#if 0
	for (int i = 0; i < tp->N (); ++i)
		if (tp->Answer (i) == 0)
			(*tp)[i][8] = -1;
#endif
#endif

	return tp;
}

