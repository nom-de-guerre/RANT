/*

Copyright (c) 2025, Douglas Santry
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

#define DEFAULT_PATH	"../../../Neural Networks/Data/MNIST/"
#define DISPLAY_VARINT(X) printf ("%s\t%d\n", #X, (int) X)
#define DISPLAY_VARFLOAT(X) printf ("%s\t%f\n", #X, (IEEE_t) X)

struct TransformerOptions_t
{
	long			to_seed;
	int				to_Nsamples;
	IEEE_t			to_haltCondition;
	int				to_maxIterations;
	int				to_Nblocks;
	int				to_Nheads;
	char			*to_path;
	bool			to_flag;
	char			*to_save;

	TransformerOptions_t (int argc, char *argv[]) :
		to_seed (time (NULL)),
		to_Nsamples (1024),
		to_haltCondition (1e-5),
		to_maxIterations (100),
		to_Nblocks (1),
		to_Nheads (4),
		to_path ((char *) DEFAULT_PATH),
		to_flag (false),
		to_save (NULL)
	{
		Parse (argc, argv);
	}

	~TransformerOptions_t (void)
	{
		if (to_save)
			free (to_save);
	}

	int Parse (int argc, char *argv[]);

	void Display (void) const
	{
		DISPLAY_VARINT (to_seed);
		DISPLAY_VARINT (to_Nsamples);
		DISPLAY_VARFLOAT (to_haltCondition);
		DISPLAY_VARINT (to_maxIterations);
	}
};

int TransformerOptions_t::Parse (int argc, char *argv[])
{
	int count = 1; // the binary's name
	char opt;

	while (true)
	{
		opt = getopt (argc, argv, "s:r:n:h:t:i:p:q");
		if (opt == -1)
			break;

		switch (opt) {
		case 'r':

			to_seed = atoi (optarg);
			count += 2;

			break;

		case 'n':

			to_Nblocks = atoi (optarg);
			count += 2;

			break;

		case 'h':

			to_Nheads = atoi (optarg);
			count += 2;

			break;

		case 't':

			to_haltCondition = atof (optarg);
			count += 2;

			break;

		case 'i':

			to_maxIterations = atoi (optarg);
			count += 2;

			break;

		case 'p':

			to_path = strdup (optarg);
			count += 2;

			break;

		case 'q':

			to_flag = true;		// App can use this for anything
			++count;

			break;

		case 's':

			to_save = strdup (optarg);
			count += 2;

			break;

		default:

			break; // we've consumed the options meant for us
		}
	}

	return count;
}

