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

#include <gradVerify.h>
#include <options.h>
#include <validate.h>

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
	int layers [] = { -1, 50, 10 };

	MNIST_t data (
		"../../../Data/MNIST/train-images.idx3-ubyte",
		"../../../Data/MNIST/train-labels.idx1-ubyte");

	Gradient_t CNN (IMAGEDIM, IMAGEDIM, 3, 10);
	CNN.setSGDSamples (params.ro_Nsamples);
	CNN.setHaltMetric (params.ro_haltCondition);
	CNN.setMaxIterations (params.ro_maxIterations);

#define NMAPS	1

	int dim = CNN.AddConvolutionLayer (NMAPS, 5, IMAGEDIM);
	dim = CNN.AddMaxPoolLayer (NMAPS, 2, dim);
	CNN.AddFullLayer (layers, Nlayers);

	for (int i = 0; i < CNN.ActiveLayers (); ++i)
		printf ("%d %s\t", CNN.LayerRows (i), (i + 1 != 3 ? " ⟶" : ""));
	printf ("\n");

	CNN.Train (data.mn_datap);

	int index = rand () % data.mn_datap->N ();
	plane_t example (IMAGEDIM, IMAGEDIM, data.mn_datap->entry (index));
	double answer = data.mn_datap->Answer (index);

	CNN.VerifyGradient (0, 1e-7, example, answer);
}

