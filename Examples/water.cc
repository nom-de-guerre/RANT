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

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <NNm.h>
#include <read_csv.h>
#include <data.h>
#include <options.h>
#include <kfold.h>

#define K_LABELS 14

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
	auto rule = (params.ro_flag ? ADAM : RPROP);

	Np = new NNet_t (layers[0] + 1, 16, K_LABELS);

	for (int i = 0; i < layers[0]; ++i)
		Np->AddDenseLayer (layers[i + 1], rule);

	Np->AddMultiCLayer (rule);

	Np->SetHalt (params.ro_haltCondition);
	Np->SetMaxIterations (params.ro_maxIterations);
	Np->SetAccuracy (); // Halt at 100% accuracy, even if above loss threshold
  	Np->SetKeepAlive (100); // Print every x epochs
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

	int N_POINTS = 20; // O->N ();
	int wrong = 0;
	IEEE_t const * Pvec;
	IEEE_t guess[K_LABELS];
	IEEE_t const * ground;

	for (int i = 0; i < N_POINTS; ++i)
	{
		int sample = rand () % O->t_N;

		Pvec = Np->ComputeVec ((*O)[sample]);
		ground = O->AnswerVec (sample);


		for (int j = 0; j < K_LABELS; ++j)
		{
			guess[j] = (Pvec[j] >= 0.5 ? 1 : 0.0);

			if (ground[j] != guess[j])
			{
				++wrong;
				accept_soln = false;
			}
		}

		printf ("Sample\t%d\n", sample);

		printf ("ground\t");
		for (int j = 0; j < K_LABELS; ++j)
			printf ("%d\t", (int) ground[j]);
		printf ("\n");

		printf ("guess\t");
		for (int j = 0; j < K_LABELS; ++j)
			printf ("%d\t", (int) guess[j]);
		printf ("\n");

printf ("\t_________________________________________________________________________________________________________\n");
	}

	if (accept_soln)
		printf (" *** Solution ACCEPTED.\n");
	else
		printf (" *** Solution REJECTED.\t%d\t%0.3f%%\n",
			wrong,
			100 * (float) wrong / (K_LABELS * (float) N_POINTS));
}

bool includeFeature [] = { false, true, true, true, true, true };

DataSet_t *LoadData (void)
{
	LoadCSV_t Z ("/Users/dsantry/Scratch/Data/water-quality-nom.csv");

	DataSet_t *tp = Z.LoadDS (30, NULL, false);
	tp->t_Nin = 16;
	tp->t_Nout = K_LABELS;

	return tp;
}

