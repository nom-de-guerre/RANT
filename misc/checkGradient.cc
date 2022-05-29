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

#include <sys/param.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <MNIST.h>

#include <gradVerify.h>
#include <options.h>

void Run (NNmConfig_t &);

int main (int argc, char *argv[])
{
	NNmConfig_t params;

	params.ro_Nsamples = 50;
	params.ro_maxIterations = 10;

	params.Parse (argc, argv);

	printf ("Seed %ld\n", params.ro_seed);

	srand (params.ro_seed);

	try {

		Run (params);

	} catch (const char *errp) {

		printf ("EXCEPTION: %s\n", errp);

	}
}

char fullpath_data [MAXPATHLEN];
char fullpath_labels [MAXPATHLEN];

void Run (NNmConfig_t &params)
{
	int Nlayers = 3;
	int layers [] = { -1, 50, 10 };

	sprintf (fullpath_data, "%s/train-images.idx3-ubyte", params.ro_path);
	sprintf (fullpath_labels, "%s/train-labels.idx1-ubyte", params.ro_path);
	MNIST_t data (fullpath_data, fullpath_labels);

	Gradient_t CNN (IMAGEDIM, IMAGEDIM, 3, 10);
	CNN.setSGDSamples (params.ro_Nsamples * data.N ());
	CNN.setHaltMetric (params.ro_haltCondition);
	CNN.setMaxIterations (5);

#define NMAPS	1

	int dim = CNN.AddConvolutionLayer (NMAPS, 3, IMAGEDIM);
	CNN.AddMaxPoolLayer (NMAPS, 2, dim);
	CNN.AddFullLayer (layers, Nlayers);

	int N = CNN.ActiveLayers ();
	for (int i = 0; i < N; ++i)
		printf ("%d %s\t", CNN.LayerRows (i), (i + 1 != N ? " ⟶" : ""));
	printf ("\n");

	CNN.Train (data.mn_datap);

	int index = rand () % data.mn_datap->N ();
	plane_t example (IMAGEDIM, IMAGEDIM, data.mn_datap->entry (index));
	double answer = data.mn_datap->Answer (index);

	printf ("Digit %d at %d\n", (int) answer, index);

	CNN.VerifyGradient (0, 1e-7, example, answer); // convolutional
	CNN.VerifyGradient (2, 1e-7, example, answer); // softmax
}

