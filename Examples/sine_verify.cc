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

/* -- example usage:

[1014]: ./sine_verify 20 10
Seed 1755611577
Input (1) ⟹ dense (20)	⟹ verify (20)	⟹ dense (10)	⟹ MSE (1)	
Loss	1.313335e-01
	Index	BPROP		Diff		Ratio
∆	0	-6.077339e-04	-6.077339e-04	0.000000
∆	1	2.691747e-03	2.691747e-03	0.000000
∆	2	2.270372e-03	2.270372e-03	0.000000
∆	3	7.215221e-04	7.215221e-04	0.000000
∆	4	3.484898e-04	3.484899e-04	0.000000
∆	5	1.822235e-03	1.822235e-03	0.000000
∆	6	1.209529e-03	1.209529e-03	0.000000
∆	7	1.149240e-03	1.149240e-03	0.000000
∆	8	-9.418498e-04	-9.418498e-04	0.000000
∆	9	1.326045e-03	1.326045e-03	0.000000
∆	10	1.432885e-03	1.432885e-03	0.000000
∆	11	1.982582e-03	1.982582e-03	0.000000
∆	12	3.937642e-03	3.937642e-03	0.000000
∆	13	4.195731e-04	4.195731e-04	0.000000
∆	14	2.256081e-03	2.256081e-03	0.000000
∆	15	4.879626e-03	4.879626e-03	0.000000
∆	16	4.791050e-03	4.791051e-03	0.000000
∆	17	2.266403e-03	2.266403e-03	0.000000
∆	18	1.085125e-03	1.085125e-03	0.000000
∆	19	3.852216e-03	3.852216e-03	0.000000

*/

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include <NNm.h>

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
	int *layers = new int [N_layers + 1];
	layers[0] = N_layers;

	for (int i = 0; i < N_layers; ++i)
		layers[i + 1] = atoi (argv[i]);

	Run (layers);

	delete [] layers;
}

void Run (int *layers)
{
	double soln_MSE = 5e-7;

	DataSet_t *O = BuildTrainingSet (N_POINTS);

	NNet_t *Np = new NNet_t (layers[0] + 2, 1, 1);

	Np->SetHalt (soln_MSE);

#define VERIFY_LAYER	1

	for (int i = 0; i < layers[0]; ++i)
	{
		if (i == VERIFY_LAYER)
			Np->AddVerificationLayer ();
		Np->AddDenseLayer (layers[i + 1], RPROP);
	}

	Np->AddScalerMSELayer (RPROP);

	Np->DisplayModel ();

	try {

		Np->Train (O, 5);

	} catch (const char *error) {

		// Ignore
	}

	printf ("Loss\t%e\n", Np->Loss ());

	auto fp = Np->DifferencingLayer (VERIFY_LAYER);

	assert (fp);

	int sample = rand () % N_POINTS;
	// middle argument is h (step size), last argument means
	// print results even if there are no errors
	fp->VerifyGradient (Np, 1e-7, (*O)[sample], true);
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

