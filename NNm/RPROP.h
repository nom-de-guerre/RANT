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

#include <stratum.h>

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
struct RPROPStrategy_t : public stratum_t
{
	NeuralM_t				r_Ei;
	NeuralM_t				r_deltaW;

	RPROPStrategy_t (const int N, const int Nin) :
		stratum_t (N, Nin),
		r_Ei (s_Nperceptrons, s_Nin),
		r_deltaW (s_Nperceptrons, s_Nin)
	{
		r_Ei.zero ();
		r_deltaW.setValue (DELTA0);
	}

	~RPROPStrategy_t (void)
	{
	}

	void Strategy (void);
	inline void RPROP (int);
};

void 
RPROPStrategy_t::Strategy (void)
{
	int Nweights = s_Nperceptrons * s_Nin;

	for (int index = 0; index < Nweights; ++index)
		RPROP (index);
}

void 
RPROPStrategy_t::RPROP (int index)
{
	double delta;
	double backtrack;

	if (r_Ei.sm_data[index] == 0.0 || s_dL.sm_data[index] == 0.0)
	{
		delta = -SIGN (s_dL.sm_data[index]) * r_deltaW.sm_data[index];

		if (isnan (delta))
			throw ("Degenerate weight update");

		s_W.sm_data[index] += delta;

		r_Ei.sm_data[index] = s_dL.sm_data[index];

	} else if (signbit (s_dL.sm_data[index]) == signbit (r_Ei.sm_data[index])) {

		// (1)
		delta = r_deltaW.sm_data[index] * ETA_PLUS;
		if (delta > DELTA_MAX)
			delta = DELTA_MAX;

		r_deltaW.sm_data[index] = delta;

		// (2)
		delta *= -(SIGN (s_dL.sm_data[index]));
		if (isnan (delta))
			throw ("Degenerate weight update");

		// (3)
		s_W.sm_data[index] += delta;

		r_Ei.sm_data[index] = s_dL.sm_data[index];

	} else {

		backtrack = r_deltaW.sm_data[index] * SIGN (r_Ei.sm_data[index]);

		// (1)
		delta = r_deltaW.sm_data[index] * ETA_MINUS;
		if (delta < DELTA_MIN)
			delta = DELTA_MIN;

		if (isnan (delta))
			throw ("Degenerate weight update");

		r_deltaW.sm_data[index] = delta;

		// (2)
		s_W.sm_data[index] += backtrack;

		// (3)
		r_Ei.sm_data[index] = 0.0;
	}

	s_dL.sm_data[index] = 0.0;

	assert (r_deltaW.sm_data[index] > 0);
}

stratum_t * AllocateRPROP (const int N, const int Nin);

stratum_t * AllocateRPROP (const int N, const int Nin)
{
	stratum_t *p = new RPROPStrategy_t (N, Nin);

	return p;
}

#define RPROP AllocateRPROP

#endif // header inclusion

