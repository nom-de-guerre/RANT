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

#include <softmaxNNm.h>
#include <options.h>

void Run (NNmConfig_t &);

int main (int argc, char *argv[])
{
	NNmConfig_t params;

	params.Parse (argc, argv);

	printf ("Seed %ld\n", params.ro_seed);

	srand (params.ro_seed);

	try {

		Run (params);

	} catch (const char *errp) {

		printf ("EXCEPTION: %s\n", errp);

	}
}

double Validate (SoftmaxNNm_t &model, DataSet_t *datap)
{
    int incorrect = 0;

    for (int i = 0; i < datap->N (); ++i)
    {
        int guess = model.Compute ((*datap)[i]);

        if (guess != datap->Answer (i))
            ++incorrect;
    }

    double ratio = (double) incorrect;
    ratio /= (double) datap->t_N;
    ratio *= 100;

    return ratio;
}

char fullpath_data [MAXPATHLEN];
char fullpath_labels [MAXPATHLEN];

void Run (NNmConfig_t &params)
{
	int Nlayers = 3;
	int layers [] = { 784, 300, 10 };

	sprintf (fullpath_data, "%s/train-images.idx3-ubyte", params.ro_path);
	sprintf (fullpath_labels, "%s/train-labels.idx1-ubyte", params.ro_path);
	MNIST_t data (fullpath_data, fullpath_labels);

	sprintf (fullpath_data, "%s/t10k-images.idx3-ubyte", params.ro_path);
	sprintf (fullpath_labels, "%s/t10k-labels.idx1-ubyte", params.ro_path);
	MNIST_t test (fullpath_data, fullpath_labels);

	double SGD = params.ro_Nsamples / (double) 100;
	SoftmaxNNm_t NNs (layers, Nlayers, RPROP);

	NNs.SetHalt (params.ro_haltCondition);
	NNs.SetAccuracy ();
	NNs.setSGD (SGD);

	try {

		NNs.Train (data.mn_datap, params.ro_maxIterations);

    } catch (const char *excep) {

        printf ("Ignore? %s\n", excep);
    }

	printf ("Verifying...\n");

	double ratio = Validate (NNs, test.mn_datap);

	printf ("%.2f%% incorrect\n", ratio);
}

