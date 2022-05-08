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

#ifndef _NN_RPROP_STRATEGY__H__
#define _NN_RPROP_STRATEGY__H__

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

// RPROP+ update parameters
#define DELTA0				1e-2
#define DELTA_MIN			1e-8
#define DELTA_MAX			50

#define ETA_PLUS			1.2
#define ETA_MINUS			0.5

#ifndef SIGN
#define SIGN(X) (signbit (X) != 0 ? -1.0 : 1.0)
#endif

/*
 * RPROP strategy implementation.
 *
 */

struct RPROP_t : public strategy_t
{
	NeuralM_t				r_Ei;
	NeuralM_t				r_delta;
	IEEE_t					*r_learnable;
	IEEE_t					*r_dL;

	RPROP_t (const int N, const int Nin, IEEE_t *W, IEEE_t *dL) :
		strategy_t (),
		r_Ei (N, Nin),
		r_delta (N, Nin),
		r_learnable (W),
		r_dL (dL)
	{
		r_Ei.zero ();
		r_delta.setValue (DELTA0);
	}

	~RPROP_t (void)
	{
	}

	virtual void _tAPI_strategy (void)
	{
		int Nweights = r_Ei.N ();

		for (int index = 0; index < Nweights; ++index)
			RPROP (index);
	}

	inline void RPROP (int);
};

void 
RPROP_t::RPROP (int index)
{
	IEEE_t delta;
	IEEE_t backtrack;

	if (r_Ei.sm_data[index] == 0.0 || r_dL[index] == 0.0)
	{
		delta = -SIGN (r_dL[index]) * r_delta.sm_data[index];

		if (isnan (delta))
			throw ("Degenerate weight update");

		r_learnable[index] += delta;

		r_Ei.sm_data[index] = r_dL[index];

	} else if (SIGN (r_dL[index]) == SIGN (r_Ei.sm_data[index])) {

		// (1)
		delta = r_delta.sm_data[index] * ETA_PLUS;
		if (delta > DELTA_MAX)
			delta = DELTA_MAX;

		r_delta.sm_data[index] = delta;

		// (2)
		delta *= -(SIGN (r_dL[index]));
		if (isnan (delta))
			throw ("Degenerate weight update");

		// (3)
		r_learnable[index] += delta;

		r_Ei.sm_data[index] = r_dL[index];

	} else {

		backtrack = r_delta.sm_data[index] * SIGN (r_Ei.sm_data[index]);

		// (1)
		delta = r_delta.sm_data[index] * ETA_MINUS;
		if (delta < DELTA_MIN)
			delta = DELTA_MIN;

		if (isnan (delta))
			throw ("Degenerate weight update");

		r_delta.sm_data[index] = delta;

		// (2)
		r_learnable[index] += backtrack;

		// (3)
		r_Ei.sm_data[index] = 0.0;
	}

	r_dL[index] = 0.0;

	assert (r_delta.sm_data[index] > 0);
}

strategy_t * AllocateRPROP (const int N, const int Nin, IEEE_t *W, IEEE_t *dL)
{
	strategy_t *p = new RPROP_t (N, Nin, W, dL);

	return p;
}

#define RPROP AllocateRPROP


#endif // header inclusion

