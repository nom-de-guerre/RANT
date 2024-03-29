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
#include <math.h>

#include <NNm.h>
#include <options.h>

#define N_POINTS	32

#define PI 			3.141592653589793
#define PI_2		1.570796326794897

#define RANGE		PI_2

DataSet_t *BuildTrainingSet (int);
void Run (NNmConfig_t &, int *);

int main (int argc, char *argv[])
{
	NNmConfig_t params;

	int consumed = params.Parse (argc, argv);

	printf ("Seed %ld\n", params.ro_seed);
	srand (params.ro_seed);

	argc -= consumed;
	argv += consumed;

	int N_layers = argc;
	// length, # inputs (1), hidden widths (argc), output width (1)
	int *layers = new int [N_layers + 1];
	layers[0] = N_layers;

	for (int i = 0; i < N_layers; ++i)
		layers[i + 1] = atoi (argv[i]);

	Run (params, layers);

	delete [] layers;
}

void Run (NNmConfig_t &params, int *layers)
{
	DataSet_t *O = BuildTrainingSet (N_POINTS);
	NNet_t *Np = NULL;
	double guess;
	auto rule = (params.ro_flag ? ADAM : RPROP);

	Np = new NNet_t (layers[0] + 1, 1, 1);

	for (int i = 0; i < layers[0]; ++i)
		Np->AddDenseLayer (layers[i + 1], rule);

	Np->AddScalerMSELayer (rule);
	Np->SetHalt (params.ro_haltCondition);
	Np->DisplayModel ();

	try {

		Np->Train (O, params.ro_maxIterations);

	} catch (const char *excep) {

		printf ("ERROR: %s\n", excep);

	}

	printf ("Finished Training: %d\t%e\n", Np->Steps (), Np->Loss ());

	int missed = 0;
	double error;
	double MSE = 0;
	double ratio;

	printf ("\t\tx\t\ty\t\tinference\terror\n");

	for (int i = 0; i < N_POINTS; ++i)
	{
		guess = Np->Compute ((*O)[i]);

		error = O->Answer(i) - guess;
		ratio = 100 * fabs (error) / O->Answer(i);
		error *= error;
		MSE += error;

		if (error > params.ro_haltCondition)
			++missed;

		printf ("DJS_RESULT\t%1.8f\t%1.8f\t%1.8f\t%f%%\t%s\n",
			(*O)[i][0],
			(*O)[i][1],
			guess,
			ratio,
			(error > params.ro_haltCondition ? "X" : ""));
	}

	MSE /= O->t_N;
	MSE *= 0.5;

	printf ("Loss\t%e in %d iterations.\n", MSE, Np->Steps ());

	if (MSE > params.ro_haltCondition)
		printf ("Accuracy not achieved: %e\t%e\n",
			MSE,
			params.ro_haltCondition);

	delete O;

	MSE = 0.0;
	O = BuildTrainingSet (64);
	for (int i = 0; i < O->t_N; ++i)
	{
		guess = Np->Compute ((*O)[i]);
		error = guess - sin ((*O)[i][0]);
		MSE += error * error;
		printf ("DJS_INFER\t%1.8f\t%1.8f\t%1.8f\n",
			(*O)[i][0],
			(*O)[i][1],
			guess);
	}

	printf ("Test MSE: %e\n", 0.5 * MSE / O->t_N);

	if (missed)
		printf ("Missed: %d\n", missed);
}

/*
 * Contruct a training set for sine that obviates the need
 * for pre-processing.  Range and domain are both (0, 1).
 *
 */

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

