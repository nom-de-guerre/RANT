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

void Run (int);

int main (int argc, char *argv[])
{
	long seed = time (0);
	printf ("Seed %ld\n", seed);

	srand (seed);

	int sample = 5000;
	if (argv[1])
		sample = atoi (argv[1]);
	Run (sample);
#if 0
	Run (2000);
	Run (4000);
	Run (8000);
	Run (10000);
#endif
}

const int LeNetC3 [] = {
1, 1, 1, 0, 0, 0,
0, 1, 1, 1, 0, 0,
0, 0, 1, 1, 1, 0,
0, 0, 0, 1, 1, 1,
1, 0, 0, 0, 1, 1,
1, 1, 0, 0, 0, 1,
1, 1, 1, 1, 0, 0,
0, 1, 1, 1, 1, 0,
0, 0, 1, 1, 1, 1,
1, 0, 0, 1, 1, 1,
1, 1, 0, 0, 1, 1,
1, 1, 1, 0, 0, 1,
1, 1, 0, 0, 1, 1,
0, 1, 1, 0, 1, 1,
1, 0, 1, 1, 0, 1,
1, 1, 1, 1, 1, 1
};

const int LeNetC5 [] = {
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

void Run (int SGD_N)
{
	int layers [] = { -1, 120, 84, 10 };

	MNIST_t data (
		"../Data/NIST/train-images.idx3-ubyte",
		"../Data/NIST/train-labels.idx1-ubyte");

	CNN_t CNN (IMAGEDIM, IMAGEDIM, 5, 10);
	CNN.setSGDSamples (SGD_N);
	CNN.setMaxIterations (10);

#define NMAPS	6
#define FSIZE	3

	int dim = CNN.AddConvolutionLayer (0, NMAPS, FSIZE, IMAGEDIM);
	dim = CNN.AddMaxPoolLayer (1, NMAPS, 2, dim);
	dim = CNN.AddConvolutionLayerStriped (2, 16, NMAPS, FSIZE, dim, LeNetC3);
	dim = CNN.AddMaxPoolLayer (3, 16, 2, dim);
	dim = CNN.AddConvolutionLayerStriped (4, 16, NMAPS, FSIZE, dim, LeNetC3);
	CNN.AddFullLayer (5, layers, 3);

	CNN.Train (data.mn_datap);

	printf ("Verifying...\n");

	int incorrect = 0;

	for (int i = 0; i < data.mn_datap->N (); ++i)
	{
		plane_t obj (IMAGEDIM, IMAGEDIM, data.mn_datap->entry (i));

		int k = CNN.Classify (&obj);
#ifdef SHOW_FILTERS
		if (i > 300 && i < 310)
		{
			obj.display ();
			CNN.DumpMaps (2);
		}
#endif
		// if (k != (*data.mn_datap)[i][IMAGEBYTES])
		if (k != data.mn_datap->Answer (i))
			++incorrect;
	}

	double ratio = (double) incorrect;
	ratio /= (double) data.mn_datap->t_N;
	ratio *= 100;

	printf ("%.2f%% incorrect\n", ratio);
}

