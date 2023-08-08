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

/*
 * The rectified linear activation function and ADAM are recommended.
 *
 */

#define __RELU

#include <MNIST.h>

#include <options.h>
#include <NNm.h>
#include <confusion.h>

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
	snprintf (fullpath_data,
		MAXPATHLEN,
		"%s/train-images.idx3-ubyte",
		params.ro_path);
	snprintf (fullpath_labels,
		MAXPATHLEN,
		"%s/train-labels.idx1-ubyte",
		params.ro_path);
	MNIST_t data (fullpath_data, fullpath_labels);

	snprintf (fullpath_data,
		MAXPATHLEN,
		"%s/t10k-images.idx3-ubyte",
		params.ro_path);
	snprintf (fullpath_labels,
		MAXPATHLEN,
		"%s/t10k-labels.idx1-ubyte",
		params.ro_path);
	MNIST_t test (fullpath_data, fullpath_labels);

	// Default to ADAM
	auto rule = (params.ro_flag ? RPROP : ADAM);

	NNet_t *Np = NULL;
	Np = new NNet_t (Nlayers + 5, IMAGEDIM * IMAGEDIM, 10);

	Np->SetHalt (params.ro_haltCondition);
	Np->SetMaxIterations (params.ro_maxIterations);
	Np->SetSGD (params.ro_Nsamples);

#define NMAPS 5

#ifdef NMAPS
	Np->Add2DFilterLayer (NMAPS, 3, 1, rule);
	Np->Add2DMaxPoolLayer (NMAPS, 2, 2);
#endif

	for (int i = 0; i < Nlayers; ++i)
		Np->AddDenseLayer (layers[i], rule);

	Np->AddSoftmaxLayer (rule);

	printf ("Learnable parameters: %d\n", Np->Nparameters ());

	Np->DisplayModel ();
	Np->DisplayShape ();

	Np->Train (data.mn_datap, params.ro_maxIterations);

	printf ("Loss\t%f\n", Np->Loss ());

	confusion_t Cm (10);
	Cm.Update (test.mn_datap, Np);

	Cm.displayInt ("Cm");
	printf ("\n\n");
	Cm.DumpStats ();

	printf ("Correct %f%%\n", 100 * Cm.ratioCorrect ());

	if (params.ro_save)
		Np->SaveModel (params.ro_save);
}

