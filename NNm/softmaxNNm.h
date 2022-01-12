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

#ifndef __NN_SOFTMAX_NNM__H__
#define __NN_SOFTMAX_NNM__H__

#include <math.h>

#include <NNm.h>
#include <softmax.h>

class SoftmaxNNm_t : public NNet_t<SoftmaxNNm_t>
{
	int						c_Correct;
	int						c_seen;
	Softmax_t				c_softm;

public:

	SoftmaxNNm_t (const int * const width,
			   const int levels,
			   stratum_t * (*alloc)(const int, const int)) :
		NNet_t (width, levels, alloc),
		c_softm (levels[width - 2] + 1, n_Nout)
	{
		_API_Cycle ();
	}

	~SoftmaxNNm_t (void)
	{
	}

	double _API_bprop (const TrainingRow_t &);
	double _API_f (double *);
	double _API_Error (DataSet_t const *);
	void _API_Cycle (void);
	bool _API_Test (DataSet_t const * const);

	double Accuracy (void) const
	{
		double right = c_Correct;
		return right / (double) c_seen;
	}

	double Loss (void) const
	{
		return n_error / (double) c_seen;
	}

	double P (int index)
	{
		return c_softm.P (index);
	}
};

double SoftmaxNNm_t::_API_f (double *x)
{
	x = n_strata[n_levels - 1]->f (x, false);

	return c_softm.ComputeSoftmax (x);
}

double SoftmaxNNm_t::_API_bprop (const TrainingRow_t &x)
{
	double loss;
	int answer = static_cast<int> (x[n_Nin]);

	int result = Compute (x); // forces computation of Softmax Pi
	if (result == answer)
		++c_Correct;
	++c_seen;

	loss = -log (c_softm.P(answer));
	if (isnan (loss) || isinf (loss))
		n_error += 1000;
	else
		n_error += loss;

	stratum_t *p = n_strata[n_levels - 1];
	stratum_t *ante = n_strata[n_levels - 2];
	double *pdL = p->s_dL.raw (); // Done here as it is row order

	assert (n_Nout == p->s_Nperceptrons);

	c_softm.bprop (answer, p->s_delta.raw (), pdL, ante->s_response.raw ());

	return loss;
}

double SoftmaxNNm_t::_API_Error (DataSet_t const * tp)
{
	return n_error / c_seen;
}

void SoftmaxNNm_t::_API_Cycle (void)
{
	n_error = 0;
	c_Correct = 0;
	c_seen = 0;
}

bool SoftmaxNNm_t::_API_Test (DataSet_t const * const tp)
{
	double Loss = _API_Error (NULL);

	return (Loss <= n_halt ? true : false);
}

#endif // header inclusion

