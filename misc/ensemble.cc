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

#include <dispatch/dispatch.h>

#include <MNIST.h>

#include <CNN.h>
#include <options.h>

void Run (RunOptions_t &);
void Verify (DataSet_t *, CNN_t **);

int main (int argc, char *argv[])
{
	RunOptions_t params;

	params.Parse (argc, argv);

	printf ("Seed %ld\n", params.ro_seed);

	srand (params.ro_seed);

	try {
		Run (params);
	} catch (const char *errp) {
		printf ("EXCEPTION: %s\n", errp);
	}
}

CNN_t *Train (DataSet_t *training, RunOptions_t &params)
{
	int layers [] = { -1, 75, 25, 10 };
	int Nlayers = 4;

	CNN_t *CNN  = new CNN_t (IMAGEDIM, IMAGEDIM, 3, 10);
	CNN->setSGDSamples (params.ro_Nsamples);
	CNN->setHaltMetric (params.ro_haltCondition);
	CNN->setMaxIterations (params.ro_maxIterations);

#define NMAPS	6

	int dim = CNN->AddConvolutionLayer (NMAPS, 3, IMAGEDIM);
	dim = CNN->AddMaxPoolSlideLayer (NMAPS, 2, dim);
	CNN->AddFullLayer (layers, Nlayers);

	CNN->Train (training);

	return CNN;
}

void Run (RunOptions_t &params)
{
	MNIST_t data (
		"../../../Data/MNIST/train-images.idx3-ubyte",
		"../../../Data/MNIST/train-labels.idx1-ubyte");
	DataSet_t *training = data.Data ();

#define N_NETS 7
	CNN_t **voters = new CNN_t *[N_NETS];
	dispatch_queue_t q = 
		dispatch_queue_create ("Neurotic", DISPATCH_QUEUE_CONCURRENT);
	dispatch_group_t jobs = dispatch_group_create ();

	for (int i = 0; i < N_NETS; ++i)
		dispatch_group_async (jobs, q, ^{ voters[i] = Train (training, params); });

	dispatch_group_wait (jobs, DISPATCH_TIME_FOREVER);

	dispatch_sync (q, ^{ Verify (training, voters); } );
}

void Verify (DataSet_t *datap, CNN_t **voters)
{
	printf ("Verifying...\n");

	int incorrect = 0;
	int votes[10];

	for (int i = 0; i < datap->N (); ++i)
	{
		memset (votes, 0, 10 * sizeof (int));
		plane_t obj (IMAGEDIM, IMAGEDIM, datap->entry (i));

		for (int j = 0; j < N_NETS; ++j)
		{
			int vote = voters[j]->Classify (&obj);

			assert (vote >= 0);
			assert (vote < 10);

			++votes[vote];
		}

		int elected = -1;
		int support = 0;
		int unique = 0;

		for (int j = 0; j < 10; ++j)
		{
			if (votes[j])
				++unique;

			if (votes[j] > support)
			{
				elected = j;
				support = votes[j];
			}
		}

		assert (elected > -1);
		assert (support > 0);

		if (elected != datap->Answer (i))
		{
			printf ("WRONG\t%d\t%d\t%d\n",
				i,
				(int) datap->Answer (i),
				unique);

			++incorrect;
		}
	}

	double ratio = (double) incorrect;
	ratio /= (double) datap->t_N;
	ratio *= 100;

	printf ("%.2f%% incorrect\n", ratio);
}

