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
#include <sys/time.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <MNIST.h>

#include <softmaxNNm.h>
#include <options.h>

void Run (NNmConfig_t &, int *);

int main (int argc, char *argv[])
{
	NNmConfig_t params;

	printf ("bytes\t%d\n", sizeof (IEEE_t));

	int consumed = params.Parse (argc, argv);

	argc -= consumed;
	argv += consumed;

	printf ("Seed %ld\n", params.ro_seed);

	srand (params.ro_seed);

	// { length, inputs,  ..., outputs }
	int *layers = new int [argc + 3];

	layers[0] = argc + 2;
	layers[1] = 784;					// 28x28 inputs
	layers[argc + 2] = 10;				// 3 outputs

	for (int i = 0; i < argc; ++i)
		layers[i + 2] = atoi (argv[i]);

	try {

		Run (params, layers);

	} catch (const char *errp) {

		printf ("EXCEPTION: %s\n", errp);

	}
}

double Validate (SoftmaxNNm_t &model, DataSet_t *datap)
{
    int incorrect = 0;

	struct timeval start, end;

	gettimeofday (&start, NULL);

    for (int i = 0; i < datap->N (); ++i)
    {
        int guess = model.Compute ((*datap)[i]);

        if (guess != datap->Answer (i))
            ++incorrect;
    }

	gettimeofday (&end, NULL);

	uint32_t a, b;
	a = 1000000 * start.tv_sec + start.tv_usec;
	b = 1000000 * end.tv_sec + end.tv_usec;

	printf ("TIME\t%f\n",
		(double) (b - a) / datap->t_N);

    double ratio = (double) incorrect;
    ratio /= (double) datap->t_N;
    ratio *= 100;

    return ratio;
}

char fullpath_data [MAXPATHLEN];
char fullpath_labels [MAXPATHLEN];

void Run (NNmConfig_t &params, int *layers)
{
	sprintf (fullpath_data, "%s/train-images.idx3-ubyte", params.ro_path);
	sprintf (fullpath_labels, "%s/train-labels.idx1-ubyte", params.ro_path);
	MNIST_t data (fullpath_data, fullpath_labels);

	sprintf (fullpath_data, "%s/t10k-images.idx3-ubyte", params.ro_path);
	sprintf (fullpath_labels, "%s/t10k-labels.idx1-ubyte", params.ro_path);
	MNIST_t test (fullpath_data, fullpath_labels);

	double SGD = 0.1; // params.ro_Nsamples / (double) 100;
	SoftmaxNNm_t NNs (layers + 1, layers[0], RPROP);

	NNs.SetHalt (params.ro_haltCondition);
	NNs.SetAccuracy ();
	NNs.SetSGD (SGD);
	NNs.SetKeepAlive (10);

	try {

		NNs.Train (data.mn_datap, params.ro_maxIterations);

    } catch (const char *excep) {

        printf ("Ignore? %s\n", excep);
    }

	printf ("Verifying...\n");

	double ratio = Validate (NNs, test.mn_datap);

	printf ("%.2f%% incorrect\n", ratio);
}

