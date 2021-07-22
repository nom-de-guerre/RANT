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

public:

	Regression_t (int *width, int levels) :
		NNet_t (width, levels)
	{
	}

	double bprop (const TrainingRow_t &);
	double f (double *);
	double Error (DataSet_t const *);

	void Cycle (void) 
	{
		n_error = 0;
	}

	bool Test (DataSet_t const * const);
};

double Regression_t::f (double *x)
{
	x = n_strata[n_levels - 1]->f (x);

	return x[0];
}

double Regression_t::bprop (const TrainingRow_t &x)
{
	double Result;
	double error = 0;

	double y = Compute (x);

	double delta;
	double dAct;
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

	return y;
}

double Regression_t::Error (DataSet_t const * tp)
{
	return n_error / tp->t_N;
}

bool Regression_t::Test (DataSet_t const * const tp)
{
	n_error /= tp->t_N;

	if (n_error <= n_halt)
		return true;

	return false;
}

#endif // header inclusion

