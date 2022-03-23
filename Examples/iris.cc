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

#include <softmaxNNm.h>
#include <read_csv.h>
#include <data.h>
#include <options.h>

void Run (NNmConfig_t &, int *);
DataSet_t *LoadData (ClassDict_t *&);

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
	int *layers = new int [N_layers + 3];

	layers[0] = N_layers + 2;
	layers[1] = 4;							// 4 inputs
	layers[N_layers + 2] = 3;				// 3 outputs

	for (int i = 0; i < N_layers; ++i)
		layers[i + 2] = atoi (argv[i]);

	Run (params, layers);

	delete [] layers;
}

#define __USE_RPROP false

void Run (NNmConfig_t &params, int *layers)
{
	ClassDict_t *dictp;
	DataSet_t *O = LoadData (dictp);
	SoftmaxNNm_t *Np = NULL;
	double guess;

	if (params.ro_flag) {

		Np = new SoftmaxNNm_t (layers + 1, layers[0], ADAM);

		printf ("Using ADAM\n");

	 } else { 

		Np = new SoftmaxNNm_t (layers + 1, layers[0], RPROP);

		printf ("Using RPROP+\n");
	}

	Np->SetHalt (params.ro_haltCondition);
	Np->SetAccuracy (); // Halt at 100% accuracy, even if above loss threshold
	Np->SetKeepAlive (50); // Print every x epochs

	try {

		Np->Train (O, params.ro_maxIterations);

	} catch (const char *excep) {

		printf ("ERROR: %s\n", excep);
	}

	printf ("\n\tLoss\t\tAccuracy\tSteps\n");
	printf ("\t%f\t%f\t%d\n\n",
		Np->Loss (),
		Np->Accuracy (),
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
				dictp->cd_dict[(int) guess].className,
				dictp->cd_dict[(int) O->Answer (i)].className,
				(correct ? ' ' : 'X'));
	}

	if (accept_soln)
		printf (" *** Solution ACCEPTED.\n");
	else
		printf (" *** Solution REJECTED.\t%f\n",
			(float) wrong / (float) N_POINTS);

}

bool includeFeature[] = { false, true, true, true, true, true };

DataSet_t *LoadData (ClassDict_t *&dictp)
{
	LoadCSV_t Z ("../../../Data/iris.csv");

	int rows;
	void *datap = Z.Load (6, rows, includeFeature);

	int stride = 4 * sizeof (double) + STR_FEATURE;

	double *table;
	dictp = ComputeClasses (
		rows,
		5,
		(const char *) datap,
		table, 
		stride);

	DataSet_t *tp = new DataSet_t (rows, 4, 1, table);

	for (int i = 0; i < 4; ++i)
	{
		tp->Center (i);
		double max = tp->Max (i);

		int N = tp->N ();
		int stride = tp->Stride ();
		double *base = (double *) tp->FeatureBase (i);

		for (int j = 0, index = 0; j < N; ++j, index += stride)
			base[index] /= max;
	}

	return tp;
}

