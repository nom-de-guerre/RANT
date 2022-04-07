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

#ifndef __NN_REGRESSION__H__
#define __NN_REGRESSION__H__

#include <NNm.h>

class Regression_t : public NNet_t<Regression_t>
{

	int			r_seen;

public:

	Regression_t (int *width,
				  int levels,
				  Rule_t alloc) :
		NNet_t (width, levels, alloc),
		r_seen (0)
	{
	}

	IEEE_t _API_bprop (const TrainingRow_t &);
	IEEE_t _API_f (IEEE_t *);
	IEEE_t _API_Error (void);

	void _API_Cycle (void) 
	{
		n_error = 0;
		r_seen = 0;
	}

	bool _API_Test (DataSet_t const * const);
};

IEEE_t Regression_t::_API_f (IEEE_t *x)
{
	x = n_strata[n_levels - 1]->f (x);

	return x[0];
}

IEEE_t Regression_t::_API_bprop (const TrainingRow_t &x)
{
	IEEE_t error = 0;

	IEEE_t y = Compute (x);

	IEEE_t delta;
	IEEE_t dAct;
	stratum_t *p = n_strata[n_levels - 1];
	stratum_t *ante = n_strata[n_levels - 2];

	error = y - x[n_Nin];
	n_error += error * error;

	dAct = DERIVATIVE_FN (y);
 
	/*
	 * ∂L   ∂y   ∂L
	 * -- · -- = -- = 𝛅 = delta
	 * ∂y   ∂∑   ∂∑
	 *
	 */

	delta = error * dAct;
	p->s_delta.sm_data[0] = delta;

	/*
	 *  ∂L   ∂y   ∂L
	 *  -- · -- = -- = 𝛅 · y(i-1)
	 *  ∂∑   ∂w   ∂w
	 *
	 */

	p->s_dL.sm_data[0] += delta;			// the bias
	for (int i = 1; i < p->s_Nin; ++i)
		p->s_dL.sm_data[i] += delta * ante->s_response.sm_data[i - 1];

	++r_seen;

	return y;
}

IEEE_t Regression_t::_API_Error (void)
{
	return n_error / r_seen;
}

bool Regression_t::_API_Test (DataSet_t const * const tp)
{
	IEEE_t Loss = _API_Error ();

	return (Loss <= n_halt ? true : false);
}

#endif // header inclusion

