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

#include <math.h>

#include <NNm.h>

#define NATURAL_NUMBER		2.718281828459045

class Softmax_t : public NNet_t<Softmax_t>
{
	double			*c_P;
	double			c_Loss;
	int				c_Correct;
	int				c_seen;

public:

	Softmax_t (const int * const width, const int levels) :
		NNet_t (width, levels)
	{
		c_P = new double [n_Nout];
		Cycle ();
	}

	~Softmax_t (void)
	{
		delete [] c_P;
	}

	double bprop (const TrainingRow_t &);
	double f (double *);
	double error (DataSet_t const *);
	void Cycle (void);
	bool Test (DataSet_t const * const);

	int ComputeSoftmax (void);

	double metric (void)
	{
		return c_Correct;
	}

	double ratio (void)
	{
		double right = c_Correct;
		return right / (double) c_seen;
	}

	double Loss (void) const
	{
		return c_Loss / (double) c_seen;
	}
};

double Softmax_t::f (double *x)
{
	for (int i = 0; i < n_Nout; ++i)
		c_P[i] = x[i];

	return ComputeSoftmax ();
}

/*
 * Convert network outputs, c_P[], to softmax "probabilities"
 *
 */
int Softmax_t::ComputeSoftmax ()
{
	double denom = 0;
	double max = -DBL_MAX;
	int factor = -1;

	for (int i = 0; i < n_Nout; ++i)
	{
		if (c_P[i] > max)
			max = c_P[i];
	}

	for (int i = 0; i < n_Nout; ++i)
	{
		c_P[i] = exp (c_P[i] - max);
		denom += c_P[i];
	}

	max = -DBL_MAX;
	for (int i = 0; i < n_Nout; ++i)
	{
		c_P[i] /= denom;

		if (c_P[i] > max)
		{
			max = c_P[i];
			factor = i;
		}
	}

	assert (max >= 0.0 && max <= 1.0);
	assert (factor > -1 && factor < n_Nout);

	return factor;
}

double Softmax_t::bprop (const TrainingRow_t &x)
{
	double loss;
	int answer = static_cast<int> (x[n_Nin]);

	int result = Compute (x); // forces computation of Softmax Pi
	if (result == answer)
		++c_Correct;
	++c_seen;

	loss = -log (c_P[answer]);
	c_Loss += loss;

	double y; 			// y = ak below
	double delta_k;
	double dAct;
	double dL;

	stratum_t *p = n_strata[n_levels - 1];
	stratum_t *ante = n_strata[n_levels - 2];
	double *pdL = p->s_dL.raw ();

	for (int output_i = 0; output_i < n_Nout; ++output_i)
	{
		y = p->s_response.sm_data[output_i];

		dL = c_P[output_i];
		if (output_i == answer)
			dL -= 1;

		/*
		 * ∂L   ∂y   ∂L
		 * -- · -- = -- = 𝛅 = delta_k
		 * ∂y   ∂∑   ∂∑
		 *
		 */

#ifdef __RELU
		if (outlayer->RELU ()
			dAct = SIGMOID_FN (outlayer->p_iprod);
		else
			dAct = DERIVATIVE_FN (y);
#else
		dAct = DERIVATIVE_FN (y);
#endif
 
		delta_k = dL * dAct;
		p->s_delta.sm_data[output_i] = delta_k;

		/*
		 * initiate the recurrence
		 *
		 * ∂L   ∂∑   ∂L
		 * -- · -- = -- = 𝛅 · y(i-1)
		 * ∂∑   ∂w   ∂w
		 *
		 */
		*pdL++ += delta_k;						// the bias
		for (int i = 1; i < p->s_Nin; ++i)
			*pdL++ += delta_k * ante->s_response.sm_data[i - 1];
	}

	return loss;
}

double Softmax_t::error (DataSet_t const * tp)
{
	return c_Loss / c_seen;
}

void Softmax_t::Cycle (void)
{
	c_Loss = 0;
	c_Correct = 0;
	c_seen = 0;
}

bool Softmax_t::Test (DataSet_t const * const tp)
{
	double ratio = c_Correct / (double) tp->t_N;

	return (ratio >= 0.95 ? true : false);
}

#endif // header inclusion

