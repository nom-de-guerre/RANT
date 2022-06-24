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

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <NNm.h>
#include <read_csv.h>
#include <data.h>
#include <options.h>

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
	double guess;
	auto rule = (params.ro_flag ? ADAM : RPROP);

	Np = new NNet_t (layers[0] + 1, 4, 3);

	for (int i = 1; i < layers[0]; ++i)
		Np->AddDenseLayer (layers[i], rule);

	Np->AddSoftmaxLayer (3, rule);

	Np->SetHalt (params.ro_haltCondition);
	Np->SetAccuracy (); // Halt at 100% accuracy, even if above loss threshold
	Np->SetKeepAlive (1); // Print every x epochs
	Np->SetNormalizePP (O);

	Np->DisplayModel ();

	try {

		Np->Train (O, params.ro_maxIterations);

	} catch (const char *excep) {

		printf ("Warning: %s\n", excep);
	}

	printf ("\n\tLoss\t\tSteps\n");
	printf ("\t%f\t%d\n\n",
		Np->Loss (),
		Np->Steps ());

	bool accept_soln = true;
	bool correct;

	printf ("\t\tTrain\tGuess\t\tCorrect\n");

	int N_POINTS = O->N ();
	int wrong = 0;

	for (int i = 0; i < N_POINTS; ++i)
	{
		guess = Np->Compute ((*O)[i]);

		correct = O->Answer (i) == guess;
		if (!correct)
		{
			accept_soln = false;
			++wrong;
		}

		if (!correct)
			printf ("(%d)\tDJS_RESULT\t%s\t%s\t%c\n",
				i,
				O->CategoryName (guess),
				O->CategoryName (O->Answer (i)),
				(correct ? ' ' : 'X'));
	}

	if (accept_soln)
		printf (" *** Solution ACCEPTED.\n");
	else
		printf (" *** Solution REJECTED.\t%f\n",
			(float) wrong / (float) N_POINTS);

}

bool includeFeature [] = { false, true, true, true, true, true };

DataSet_t *LoadData (void)
{
	LoadCSV_t Z ("../../../Neural Networks/Data/iris.csv");

	DataSet_t *tp = Z.LoadDS (6, includeFeature);

	return tp;
}

