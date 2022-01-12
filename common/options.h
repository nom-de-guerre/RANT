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
#include <unistd.h>
#include <string.h>

#define DEFAULT_PATH	"../../../Data/MNIST/"
#define DISPLAY_VARINT(X) printf ("%s\t%d\n", #X, (int) X)
#define DISPLAY_VARFLOAT(X) printf ("%s\t%f\n", #X, (double) X)

struct NNmConfig_t
{
	long			ro_seed;
	int				ro_Nsamples;
	double			ro_haltCondition;
	int				ro_maxIterations;
	char			*ro_path;

	NNmConfig_t () :
		ro_seed (time (NULL)),
		ro_Nsamples (1000),
		ro_haltCondition (1e-5),
		ro_maxIterations (100),
		ro_path ((char *) DEFAULT_PATH)
	{}

	int Parse (int argc, char *argv[]);

	void Display (void) const
	{
		DISPLAY_VARINT (ro_seed);
		DISPLAY_VARINT (ro_Nsamples);
		DISPLAY_VARFLOAT (ro_haltCondition);
		DISPLAY_VARINT (ro_maxIterations);
	}
};

int NNmConfig_t::Parse (int argc, char *argv[])
{
	int count = 1; // the binary's name
	char opt;

	while (true)
	{
		opt = getopt (argc, argv, "s:n:t:i:h");
		if (opt == -1)
			break;

		switch (opt) {
		case 's':

			ro_seed = atoi (optarg);
			count += 2;

			break;

		case 'n':

			ro_Nsamples = atoi (optarg);
			count += 2;

			break;

		case 't':

			ro_haltCondition = atof (optarg);
			count += 2;

			break;

		case 'i':

			ro_maxIterations = atoi (optarg);
			count += 2;

			break;

		case 'p':

			ro_path = strdup (optarg);
			count += 2;

			break;

		case 'h':

			printf ("usage: %s [-i iterations] [-s seed] [-n samples] [-t halt condition] [application options]\n",
				argv[0]);
			exit (-1);

		default:

			break; // we've consumed the options meant for us
		}
	}

	return count;
}

