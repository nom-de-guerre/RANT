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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <MNIST.h>

#include <options.h>
#include <NNm.h>
#include <NNm_verify.h>

#include <plane.h>

void Run (NNmConfig_t &, const int, int *);

int main (int argc, char *argv[])
{
	printf ("IEEE_t\t%lu\n", sizeof (IEEE_t));

	NNmConfig_t params;

	int consumed = params.Parse (argc, argv);

	printf ("Seed %ld\n", params.ro_seed);
	srand (params.ro_seed);

	argc -= consumed;
	argv += consumed;

#define MAX_ENTRIES 16
	int Nlayers = 3;
	int layers [MAX_ENTRIES];

	layers[0] = -1;

	Nlayers = argc;
	for (int i = 0; i < argc; ++i)
		layers[i] = atoi (argv[i]);

	try {

		Run (params, Nlayers, layers);

	} catch (const char *errp) {

		printf ("EXCEPTION: %s\n", errp);
	}
}

char fullpath_data [MAXPATHLEN];
char fullpath_labels [MAXPATHLEN];

void Run (NNmConfig_t &params, const int Nlayers, int *layers)
{
	sprintf (fullpath_data, "%s/train-images.idx3-ubyte", params.ro_path);
	sprintf (fullpath_labels, "%s/train-labels.idx1-ubyte", params.ro_path);
	MNIST_t data (fullpath_data, fullpath_labels);

	sprintf (fullpath_data, "%s/t10k-images.idx3-ubyte", params.ro_path);
	sprintf (fullpath_labels, "%s/t10k-labels.idx1-ubyte", params.ro_path);
	MNIST_t test (fullpath_data, fullpath_labels);

	auto rule = (params.ro_flag ? ADAM : RPROP);

#ifdef __VERIFY_GRAD
	VerifyGrad_t *Np = NULL;
	Np = new VerifyGrad_t (Nlayers + 3, IMAGEDIM * IMAGEDIM, 10);
#else
	NNet_t *Np = NULL;
	Np = new NNet_t (Nlayers + 5, IMAGEDIM * IMAGEDIM, 10);
#endif

	Np->SetHalt (params.ro_haltCondition);
	Np->SetMaxIterations (params.ro_maxIterations);
	// Np->SetKeepAlive (10);
	Np->SetSGD (params.ro_Nsamples);

#define NMAPS	10

//	Np->AddConvolutionLayer (NMAPS, 3, rule);
//	Np->AddMaxPoolLayer (NMAPS, 2);

	for (int i = 0; i < Nlayers; ++i)
		Np->AddDenseLayer (layers[i], rule);

	Np->AddSoftmaxLayer (rule);

	printf ("Learnable parameters: %d\n", Np->Nparameters ());

	Np->DisplayModel ();
	Np->DisplayShape ();

	Np->Train (data.mn_datap, params.ro_maxIterations);

#if 0
	int EXAMPLE = rand () % 60000;

	plane_t obj (IMAGEDIM, IMAGEDIM, data.mn_datap->entry (EXAMPLE));
	obj.displayImage ("");

	Np->Compute ((*data.mn_datap)[EXAMPLE]);

	Np->DumpMaps (0);
	Np->DumpMaps (1);

#endif

#ifdef __VERIFY_GRAD
	Np->VerifyGradient (0, 1e-7, (*data.mn_datap)[5247]);
#endif

	printf ("Loss\t%f\n", Np->Loss ());

	printf ("Verifying...\n");

	int wrong = 0;
	for (int i = 0; i < data.mn_datap->N (); ++i)
	{
		int guess = (int) Np->Compute ((*data.mn_datap)[i]);
		if (guess != data.mn_datap->Answer (i))
			++wrong;
	}

	double ratio = (double) wrong / data.mn_datap->N ();
	printf ("%.2f%% incorrect\n", 100 * ratio);
}

