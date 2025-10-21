
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

#include <stdio.h>
#include <readline/readline.h>
#include <readline/history.h>

#include <TextData.h>
#include <CausalModel.h>
#include <Toptions.h>

void Interaction (CausalModel_t &NLM, CausalData_t &data, int N=1024);

int main (int argc, char *argv[])
{
	TransformerOptions_t opts (argc, argv);

	long seed = opts.to_seed;

	printf ("Using seed %ld\n", seed);
	srand (seed);

	printf ("IEEE_t is %lu bytes.\n", sizeof (IEEE_t));
#ifdef __POSITIONAL
	printf ("Including Positional Embeddings\n");
#endif
#ifndef __NO_E_LEARNING
    printf ("Including Learned Embeddings\n");
#endif
#ifndef __NO_TRANSBLOCK_FFN
    printf ("Including FFN\n");
#endif
#ifndef __NO_LAYERNORM_
    printf ("Including LN\n");
#endif

	if (opts.to_flag)
		printf ("Running with learnable E.\n");

	CausalData_t data (
		"Data/SherlockHolmesNormalized.txt",
		"Data/Sherlock.E",
		opts.to_flag);

	printf ("\t\t\t\tBlocks\tHeads\tWindow\td\n");
	printf ("Running with transformer\t%d\t%d\t%d\t%d\n",
		opts.to_Nblocks,
		opts.to_Nheads,
		TOKENWINDOW,
		data.get_d ());

	CausalModel_t NLM (opts.to_Nblocks,
						opts.to_Nheads,
						TOKENWINDOW,
						data.get_d (),
						data.getV_N ());

	printf ("Model has %d learnable parameters.\n",
		NLM.N_LearnableParameters () + data.N_LearnableParameters ());

	printf ("Final Loss %f\n",
			NLM.fit (data,
				opts.to_Nsamples,
				opts.to_maxIterations));

	Interaction (NLM, data, opts.to_Nsamples);
}

void Interaction (CausalModel_t &NLM, CausalData_t &data, int N)
{
	bool rc = true;
	char *line;
	Md_t X;
	int const *y;

	while (rc)
	{
		line = readline ("-> ");
		if (line == NULL)
			break;

		if (line[0] == '$')
		{
			if (strncmp (line + 1, "list", 4) == 0)
			{
				data.reset ();

				for (int i = 0; i < N; ++i)
				{
					int Ntokens = data.nextClause ();
					for (int i = 0; i < Ntokens; ++i)
						printf ("%s ",
							(char const *) data.cd_lexemes[i].iov_base);
					printf ("\n");
				}

			} else if (strncmp (line + 1, "bye", 3) == 0) {

				free (line);
				return;

			} else if (strncmp (line + 1, "save", 4) == 0) {

				char *p = line + 6;
				if (strncmp (p, "model", 5) == 0)
					NLM.save (p + 6);
				else if (strncmp (p, "V", 1) == 0)
					data.cd_V.save (p + 2);
				else if (strncmp (p, "all", 3)) {
					char buffer[128];
					snprintf (buffer, 128, "%s.E", p+4);
					data.cd_V.save (buffer);
					snprintf (buffer, 128, "%s.RANT", p+4);
					NLM.save (buffer);
				} else
					printf ("Save command not recognized\n");
			} else
				printf ("Command not recognized\n");

			free (line);
			continue;
		}

		char *lineHistory = strdup (line);

		data.cd_V.StringToX (line, X, true);

		y = NLM.predict (X);
		data.cd_V.printTokens (X.rows (), y);

		free (line);

		add_history (lineHistory);
	}
}

