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

#ifndef _NN_MSE_REGRESSION__H__
#define _NN_MSE_REGRESSION__H__

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include <NNm.h>

/*
 * Implements a layer when training a neural network.
 *
 */
struct ScalerMSE_t : public stratum_t
{
	NeuralM_t				ms_W;
	NeuralM_t				ms_dL;

	ScalerMSE_t (const int ID, const int Nin, StrategyAlloc_t rule) :
		stratum_t ("MSE", ID, 1, Nin + 1),
		ms_W (1, Nin + 1),
		ms_dL (1, Nin + 1)
	{
		s_strat = (*rule) (1, Nin + 1, ms_W.raw (), ms_dL.raw ());
	}

	virtual ~ScalerMSE_t (void)
	{
	}

	void _sAPI_init (void)
	{
		for (int i = 0; i < s_Nin; ++i)
			ms_W.sm_data[i] = ((IEEE_t) rand () / RAND_MAX) - 0.5;

		ms_dL.zero ();
	}

	virtual IEEE_t * _sAPI_f (IEEE_t * const, bool = true);
	virtual void _sAPI_gradient (stratum_t &);
	virtual void _sAPI_bprop (IEEE_t *, bool = true);
	virtual IEEE_t _sAPI_Loss (IEEE_t const * const);

	virtual int _sAPI_Trainable (void)
	{
		return ms_W.N ();
	}

	virtual void StrategyMono (const int index)
	{
		return; // over-ride when debugging or instrumenting
	}
};

void
ScalerMSE_t::_sAPI_gradient (stratum_t &Z)
{
	Z.s_delta.TransposeMatrixVectorMult (ms_W, s_delta.raw ());
}

void 
ScalerMSE_t::_sAPI_bprop (IEEE_t *xi, bool activation)
{
	s_delta.sm_data[0] *= DERIVATIVE_FN (s_response.sm_data[0]);

	IEEE_t *dL = ms_dL.sm_data;
	IEEE_t delta = s_delta.sm_data[0];

	*dL++ += delta; // the Bias

	for (int i = 1; i < s_Nin; ++i)
		*dL++ += delta * xi[i - 1];
}

IEEE_t *
ScalerMSE_t::_sAPI_f (IEEE_t * const xi, bool activate)
{
	s_response.MatrixVectorMult (ms_W, xi);

	IEEE_t *f = s_response.raw ();

	if (activate)
		for (int i = 0; i < s_Nin; ++i, ++f)
			*f = ACTIVATION_FN (*f);

	return s_response.raw ();
}

IEEE_t
ScalerMSE_t::_sAPI_Loss (IEEE_t const * const answers)
{
	IEEE_t loss = 0.0;

	loss = s_delta.sm_data[0] = s_response.sm_data[0] - answers[0];
	loss *= loss;

	return 0.5 * loss;
}

#endif // header inclusion

