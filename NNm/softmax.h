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

#ifndef __NN_SOFTMAX__H__
#define __NN_SOFTMAX__H__

#include <math.h>

#include <NNm.h>

class Softmax_t
{
	const int		so_Nclasses;
	IEEE_t			*so_P;

public:

	Softmax_t (const int Nclasses) :
		so_Nclasses (Nclasses),
		so_P (new IEEE_t [so_Nclasses])
	{
	}

	~Softmax_t (void)
	{
		delete [] so_P;
	}

	int ComputeSoftmax (IEEE_t const * const);
	void bprop (const int, IEEE_t *);

	IEEE_t P (int x)
	{
		return so_P[x];
	}
};

/*
 * Convert network outputs, so_P[], to softmax "probabilities"
 *
 */
int Softmax_t::ComputeSoftmax (IEEE_t const * const Xi)
{
	IEEE_t denom = 0;
	IEEE_t max = -DBL_MAX;
	int factor = -1;

	for (int i = 0; i < so_Nclasses; ++i)
	{
		if (Xi[i] > max)
			max = Xi[i];
	}

	for (int i = 0; i < so_Nclasses; ++i)
	{
		so_P[i] = exp (Xi[i] - max);
		denom += so_P[i];
	}

	max = -DBL_MAX;
	for (int i = 0; i < so_Nclasses; ++i)
	{
		so_P[i] /= denom;

		if (so_P[i] > max)
		{
			max = so_P[i];
			factor = i;
		}
	}

	assert (max >= 0.0 && max <= 1.0);
	assert (factor > -1 && factor < so_Nclasses);

	return factor;
}

void Softmax_t::bprop ( const int answer, IEEE_t * deltap)
{
	IEEE_t dL;

	for (int output_i = 0; output_i < so_Nclasses; ++output_i)
	{
		dL = so_P[output_i];
		if (output_i == answer)
			dL -= 1;

		/*
		 * ∂L
		 * -- = q - p = 𝛅 = dL
		 * ∂y
		 *
		 */

		deltap[output_i] = dL;
	}
}

#endif // header inclusion

