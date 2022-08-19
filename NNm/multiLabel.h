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

#ifndef __NNm_MULTILABEL__H__
#define __NNm_MULTILABEL__H__

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
struct multiL_t : public stratum_t
{
	// Matrices - per weight, s_Nnodes x s_Nin
	NeuralM_t				mx_W;
	NeuralM_t				mx_dL;

	// Vectors - per perceptron (node)
	NeuralM_t				mx_dot;			// Wx

	multiL_t (const int ID, const int N, const int Nin, StrategyAlloc_t rule) :
		stratum_t ("multiL", ID, N, Nin + 1),	// account for bias
		mx_W (N, Nin + 1),
		mx_dL (N, Nin + 1),
		mx_dot (N, 1)
	{
		s_strat = (*rule) (N, Nin + 1, mx_W.raw (), mx_dL.raw ());
	}

	multiL_t (const int N, const int Nin, StrategyAlloc_t rule) :
		stratum_t ("dense", -1, N, Nin + 1),	// account for bias
		mx_W (N, Nin + 1),
		mx_dL (N, Nin + 1),
		mx_dot (N, 1)
	{
		s_strat = (*rule) (N, Nin + 1, mx_W.raw (), mx_dL.raw ());
	}

	virtual ~multiL_t (void)
	{
	}

	void _sAPI_init (void)
	{
		int Nout = mx_W.rows ();

		mx_dL.zero ();

		// Glorot, W ~ [-r, r]
		IEEE_t r = sqrt (6.0 / (Nout + s_Nin));
		IEEE_t *p = mx_W.raw();
		IEEE_t sample;

		for (int i = mx_W.rows () - 1; i >= 0; --i)
			for (int j = mx_W.columns () - 1; j >= 0; --j)
			{
				sample = (IEEE_t) rand () / RAND_MAX;
				sample *= r;
				if (rand () % 2)
					sample = -sample;
				*p++ = sample;
			}
	}

	virtual IEEE_t * _sAPI_f (IEEE_t * const, bool = true);
	virtual void _sAPI_gradient (stratum_t &);
	virtual void _sAPI_bprop (IEEE_t *, bool = true);

	virtual int _sAPI_Trainable (void)
	{
		return mx_W.N ();
	}

	virtual IEEE_t _sAPI_Loss (IEEE_t const * const answers);

	virtual void StrategyMono (const int index)
	{
		return; // over-ride when debugging or instrumenting
	}
};

void
multiL_t::_sAPI_gradient (stratum_t &Z)
{
	Z.s_delta.TransposeMatrixVectorMult (mx_W, s_delta.raw ());
}

void 
multiL_t::_sAPI_bprop (IEEE_t *xi, bool activation)
{
	/*
	 *
	 * We are called with,
	 *
	 *    ∂L    ∂L   ∂p
	 *    --  = -- · --
	 *    ∂u    ∂p   ∂u
	 *
	 * in s_delta.  The sigmoid derivative cancels out the denom. in loss
	 * derivative.
	 *
	 */

	// Apply the delta for per weight derivatives

	IEEE_t *dL = mx_dL.sm_data;
	IEEE_t delta;

	/*
	 * ∂L       ∂∑
	 * -- = 𝛿 · -- = 𝛿 · Xi
	 * ∂w       ∂w
	 *
	 * ∆W = 𝛿 · transpose (Xi), this is an outer product, but we do 
	 * it here instead of NeuralM.
	 *
	 */
	for (int i = 0; i < s_Nnodes; ++i)
	{
		delta = s_delta.sm_data[i];
		*dL++ += delta; // the Bias

		for (int j = 1; j < s_Nin; ++j)
			*dL++ += delta * xi[j - 1];
	}
}

IEEE_t *
multiL_t::_sAPI_f (IEEE_t * const xi, bool activate)
{
	mx_dot.MatrixVectorMult (mx_W, xi);

	IEEE_t *p = s_response.sm_data;
	IEEE_t *dot = mx_dot.sm_data;

	for (int i = 0; i < s_Nnodes; ++i)
		*p++ = SIGMOID_FN (*dot++);

	return s_response.sm_data;
}

IEEE_t
multiL_t::_sAPI_Loss (IEEE_t const * const answers)
{
	IEEE_t *delta = s_delta.raw ();
	IEEE_t *_p = s_response.raw ();
	IEEE_t loss = 0.0;

	for (int i = 0; i < s_Nnodes; ++i)
	{
		delta[i] = _p[i] - answers[i];
		if (answers[i])
			loss += -log (_p[i]);
		else
			loss += -log (1 - _p[i]);
// printf ("(%f, %f)\t", _p[i], answers[i]);
	}
// printf ("\n");

	loss /= s_Nnodes;

	return loss;
}

#endif // header inclusion

