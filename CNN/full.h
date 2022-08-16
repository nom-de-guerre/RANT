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

#ifndef __DJS_FULL__H__
#define __DJS_FULL__H__

#include <string.h>

#include <ANT.h>

#include <NNm.h>
#include <layer.h>

class full_t : public NNet_t, public mapAPI_t
{
	int				re_Nin;

	IEEE_t			*re_input;
	NeuralM_t		re_gradient;

public:

	full_t (const int * const layers,
			const int Nlayers,
			const int Nin,				// Number of inputs
			const int K,				// Number of categories
			StrategyAlloc_t rule) :
		NNet_t (Nlayers + 1, Nin, K),
		mapAPI_t (Nin),
		re_Nin (Nin),
		re_input (new IEEE_t [re_Nin + 1]), // + 1 for training - the answer
		re_gradient (re_Nin, 1, ma_map.raw ())
	{
		for (int i = 0; i < Nlayers; ++i)
			AddDenseLayer (layers[i], rule);

		AddSoftmaxLayer (rule);

		printf ("ANN: ");
		DisplayModel ();
	}

	~full_t (void)
	{
		delete [] re_input;
	}

	bool Forward (arg_t &arg)
	{
		int len = Load (arg);
		assert (len == re_Nin);

		ma_signal = (int) Compute (re_input);

		return true;
	}

	int Load (arg_t &arg)
	{
		int blockSize = arg.a_args[0]->N ();
		int index = 0;

		for (int i = 0; i < arg.a_N; ++i, index += blockSize)
			memcpy (re_input + index, 
				arg.a_args[i]->raw (), 
				blockSize * sizeof (IEEE_t));

		return index;
	}

	bool Train (arg_t &arg, IEEE_t answer)
	{
		int len = Load (arg);
		assert (len == re_Nin);

		re_input[len] = answer;

		ComputeDerivative (re_input);
		re_gradient.TransposeMatrixVectorMult (
			static_cast<dense_t *> (n_strata[0])->de_W,
			n_strata[0]->s_delta.raw ());

		return true;
	}

	bool Backward (arg_t &arg)
	{
		return true;
	}

	bool Update (void)
	{
		UpdateWeights ();
printf ("UPDATE\t%e\n", n_error);
		n_error = 0;

		return true;
	}

	plane_t *fetchGradient (void)
	{
		return &ma_map;
	}
};

#endif // header inclusion

