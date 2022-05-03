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

#include <NNm_verify.h>

#define N_POINTS	32

#define PI 			3.141592653589793
#define PI_2		1.570796326794897
#define PI_DELTA	(PI_2 / N_POINTS)

DataSet_t *BuildTrainingSet (int);
void Run (int *);

int main (int argc, char *argv[])
{
	if (argc < 2)
	{
		printf ("Usage: list of hidden-layers\n");
		exit (-1);
	}

	long seed = time (0);
	printf ("Seed %ld\n", seed);

	srand (seed);

	--argc;
	++argv;

	int N_layers = argc;
	int *layers = new int [N_layers + 3];	// widths plus length prefix, inputs
	layers[0] = N_layers + 2;
	layers[1] = 1;							// one input
	layers[N_layers + 2] = 1;				// one output

	for (int i = 0; i < N_layers; ++i)
		layers[i + 2] = atoi (argv[i]);

	Run (layers);

	delete [] layers;
}

void Run (int *layers)
{
	double soln_MSE = 5e-7;

	DataSet_t *O = BuildTrainingSet (N_POINTS);
	RegressionGrad_t *Np = NULL;

	Np = new RegressionGrad_t (layers + 1, layers[0]);
	Np->SetHalt (soln_MSE);

	try {

		Np->Train (O, 10);

	} catch (const char *error) {

		// Ignore
	}

	// We only examine the input layer and the hidden layers
	for (int i = 0; i < layers[0] - 2; ++i)
	{
		int sample = rand () % N_POINTS;
		Np->VerifyGradient (i, 1e-7, (*O)[sample]);
	}
}

DataSet_t *BuildTrainingSet (int N)
{
	DataSet_t *O = new DataSet_t (N, 1, 1);

	for (int i = 0; i < N; ++i)
	{
		double sample = (double) rand () / RAND_MAX;

		(*O)[i][0] = sample * PI_2;
		(*O)[i][1] = sin ((*O)[i][0]);
	}

	return O;
}

