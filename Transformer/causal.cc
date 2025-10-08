
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

#include <TextData.h>
#include <CausalModel.h>
#include <Toptions.h>

int main (int argc, char *argv[])
{
	TransformerOptions_t opts (argc, argv);

	long seed = opts.to_seed;

	printf ("Using seed %ld\n", seed);
	srand (seed);

	printf ("IEEE_t is %lu bytes.\n", sizeof (IEEE_t));
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
}

