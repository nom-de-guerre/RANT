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
 * Running as, ./iris -i 7500 20 20, should converge.
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
#include <kfold.h>

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

	layers[0] = N_layers;

	for (int i = 0; i < N_layers; ++i)
		layers[i + 1] = atoi (argv[i]);

	Run (params, layers);

	delete [] layers;
}

void Run (NNmConfig_t &params, int *layers)
{
	DataSet_t *O = LoadData ();
	NNet_t *Np = NULL;
	auto rule = (params.ro_flag ? RPROP : ADAM);

	Np = new NNet_t (layers[0] + 2, 4, 3);

	Np->AddPreProcessingLayer (O);

	for (int i = 0; i < layers[0]; ++i)
		Np->AddDenseLayer (layers[i + 1], rule);

	Np->AddSoftmaxLayer (rule);

	Np->SetHalt (params.ro_haltCondition);
	Np->SetMaxIterations (params.ro_maxIterations);
	Np->SetAccuracy (); // Halt at 100% accuracy, even if above loss threshold

	Np->DisplayModel ();
	printf ("Learnable parameters %d\n", Np->Nparameters ());

#ifdef __FOLDED_RUN

	confusion_t Cm (3);
	kfold_t kf (O);

	kf.ValidateConfM (Np, 3, Cm);

	Cm.display ();

	return;

#else

	try {

		Np->Train (O, params.ro_maxIterations);

	} catch (const char *excep) {

		printf ("Warning: %s\n", excep);
	}

	printf ("\n\tLoss\t\tSteps\n");
	printf ("\t%f\t%d\n\n",
		Np->Loss (),
		Np->Steps ());

	double guess;
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

#endif // __FOLDED_RUN
}

bool includeFeature [] = { true, true, true, true, true };

DataSet_t *LoadData (void)
{
	LoadCSV_t Z ("../Data/iris.csv");

	DataSet_t *tp = Z.LoadDS (5, includeFeature);

	return tp;
}

