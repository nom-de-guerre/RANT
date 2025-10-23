
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

void Interaction (CausalModel_t &NLM, vocabDict_t &data, int N=1024);

char buffer[128];

int main (int argc, char *argv[])
{
	TransformerOptions_t opts (argc, argv);

	vocabDict_t E (opts.to_Efile);

	FILE *fp = fopen (opts.to_file, "r");
	if (fp == NULL)
	{
		printf ("Unable to open %s, %s\n", opts.to_file, strerror (errno));
		exit (-1);
	}

	int rc = fscanf (fp, "%s\n", buffer);
	if (rc < 0)
	{
		printf ("Unable to read %s, %s\n", opts.to_file, strerror (errno));
		exit (-1);
	}

	CausalModel_t NLM (fp);

	fclose (fp);

	Interaction (NLM, E, opts.to_Nsamples);
}

void Interaction (CausalModel_t &NLM, vocabDict_t &data, int N)
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
			if (strncmp (line + 1, "bye", 3) == 0) {

				free (line);
				return;

			} else
				printf ("Command not recognized\n");

			free (line);
			continue;
		}

		char *lineHistory = strdup (line);

		data.StringToX (line, X, true);

		y = NLM.predict (X);
		data.printTokens (X.rows (), y);

		free (line);

		add_history (lineHistory);
	}
}

