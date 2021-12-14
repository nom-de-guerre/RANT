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

void Run (int *);
DataSet_t *LoadData (void);

int main (int argc, char *argv[])
{
	if (argc < 2)
	{
		printf ("Usage: LoW hidden-layers output-layers\n");
		exit (-1);
	}

	long seed = time (0);
	printf ("Seed %ld\n", seed);

	srand (seed);

	int N_layers = argc - 1;

	// widths plus length prefix, inputs
	int *layers = new int [N_layers + 2];

	layers[0] = N_layers + 1;
	layers[1] = 4;							// 4 input
	for (int i = 0; i < N_layers; ++i)
		layers[i + 2] = atoi (argv[i + 1]);

	Run (layers);

	delete [] layers;
}

void Run (int *layers)
{
	DataSet_t *O = LoadData ();
	SoftmaxNNm_t *Np = NULL;
	double guess;

	Np = new SoftmaxNNm_t (layers + 1, layers[0], ADAM);

	Np->SetHalt (1e-2);

	try {

		Np->Train (O, 20000);

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

	printf ("\t\tTrain\tGuess\tCorrect\n");

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
			printf ("(%d)\tDJS_RESULT\t%d\t%d\t%c\n",
				i,
				(int) O->Answer (i),
				(int) guess,
				(correct ? ' ' : 'X'));
	}

	if (accept_soln)
		printf (" *** Solution ACCEPTED.\n");
	else
		printf (" *** Solution REJECTED.\t%f\n",
			(float) wrong / (float) N_POINTS);

}


bool includeFeature[] = { false, true, true, true, true, true };

DataSet_t *LoadData ()
{
	LoadCSV_t Z ("../../../Data/iris.csv");

	int rows;
	void * datap = Z.Load (6, rows, includeFeature);

	int stride = 4 * sizeof (double) + STR_FEATURE;

	char *startp = (char *) datap;
	startp += 4 * sizeof (double);

	double *table;
	// ClassDict_t *dictp = ComputeClasses (
	(void) ComputeClasses (
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

