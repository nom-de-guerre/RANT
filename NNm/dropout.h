/*

Copyright (c) 2022, Douglas Santry
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

#ifndef _NNm_DROPOUT__H__
#define _NNm_DROPOUT__H__

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include <NeuralM.h>
#include <NNm.h>

/*
 * Implements a layer when training a neural network.
 *
 */
struct dropout_t : public stratum_t
{
	IEEE_t					do_pRetain;

	dropout_t (const int N, IEEE_t p_retain) : 
		stratum_t ("dropout", -1, N, N),	// account for bias
		do_pRetain (p_retain)
	{
	}

	virtual ~dropout_t (void)
	{
	}

	void _sAPI_init (void)
	{
	}

	virtual IEEE_t * _sAPI_f (IEEE_t * const, bool = true);
	virtual void _sAPI_gradient (stratum_t &);
	virtual void _sAPI_bprop (IEEE_t *, bool = true);

	virtual void StrategyMono (const int index)
	{
		return; // over-ride when debugging or instrumenting
	}
};

void
dropout_t::_sAPI_gradient (stratum_t &Z)
{
	for (int i = 0; i < s_Nin; ++i)
		if (s_response (i, 0) != 0.0)
			Z.s_delta (i, 0) = s_delta (i, 0);
		else
			Z.s_delta (i, 0) = 0.0;
}

void 
dropout_t::_sAPI_bprop (IEEE_t *xi, bool activation)
{
	// Nothing to do, no learnable parameters.
}

IEEE_t *
dropout_t::_sAPI_f (IEEE_t * const xi, bool activate)
{
	IEEE_t *p = s_response.raw ();

	if (s_frozen) // normal inference
	{
		for (int i = 0; i < s_Nin; ++i)
			p[i] = do_pRetain * xi[i];

	} else {

		IEEE_t sample;

		for (int i = 0; i < s_Nin; ++i)
		{
			sample = (IEEE_t) rand () / RAND_MAX;

			p[i] = (sample > do_pRetain ? 0.0 : xi[i]);
		}
	}

	return s_response.sm_data;
}

#endif // header inclusion

