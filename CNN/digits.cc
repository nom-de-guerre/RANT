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

#include <MNIST.h>

#include <CNN.h>
#include <options.h>

void Run (RunOptions_t &);

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

void Run (RunOptions_t &params)
{
	int Nlayers = 3;
	int layers [] = { -1, 20, 10 };

	MNIST_t data (
		"../../../Data/NIST/train-images.idx3-ubyte",
		"../../../Data/NIST/train-labels.idx1-ubyte");

	CNN_t CNN (IMAGEDIM, IMAGEDIM, 3, 10);
	CNN.setSGDSamples (params.ro_Nsamples);
	CNN.setHaltMetric (params.ro_haltCondition);
	CNN.setMaxIterations (params.ro_maxIterations);

#define NMAPS	6

	int dim = CNN.AddConvolutionLayer (NMAPS, 3, IMAGEDIM);
	dim = CNN.AddMaxPoolLayer (NMAPS, 3, dim);
	CNN.AddFullLayer (layers, Nlayers);

	CNN.Train (data.mn_datap);

	printf ("Training halted after %d iterations.\n", CNN.Steps ());

	printf ("Verifying...\n");

	int incorrect = 0;

	int base = rand () % data.N ();
	if (base + 10 > data.N ())
		base -= 10 + rand () % 100;

	for (int i = 0; i < data.mn_datap->N (); ++i)
	{
		plane_t obj (IMAGEDIM, IMAGEDIM, data.mn_datap->entry (i));

		int k = CNN.Classify (&obj);
#ifdef SHOW_FILTERS
		if (i >= base && i < (base + 10))
		{
			obj.display ();
			CNN.DumpMaps (0);
			CNN.DumpMaps (1);
		}
#endif
		if (k != data.mn_datap->Answer (i))
			++incorrect;
	}

	double ratio = (double) incorrect;
	ratio /= (double) data.mn_datap->t_N;
	ratio *= 100;

	printf ("%.2f%% incorrect\n", ratio);
}

