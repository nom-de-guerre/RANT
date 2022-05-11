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

#ifndef _NN_DENSE__H__
#define _NN_DENSE__H__

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include <NeuralM.h>
#include <NNm.h>

#define RECTIFIER(X) log (1 + exp (X))
#define SIGMOID_FN(X) (1 / (1 + exp (-X))) // derivative of rectifier

#ifdef __TANH_ACT_FN
#define ACTIVATION_FN(X) tanh(X)
#define DERIVATIVE_FN(Y) (1 - Y*Y)
#else
#define ACTIVATION_FN(X) SIGMOID_FN(X)
#define DERIVATIVE_FN(Y) (Y * (1 - Y))
#endif

/*
 * Implements a layer when training a neural network.
 *
 */
struct dense_t : public stratum_t
{
	// Matrices - per weight, s_Nnodes x s_Nin
	NeuralM_t				de_W;
	NeuralM_t				de_dL;

	// Vectors - per perceptron (node)
	NeuralM_t				de_dot;			// Wx

	dense_t (const int N, const int Nin, StrategyAlloc_t rule) :
		stratum_t (N, Nin + 1),				// account for bias
		de_W (N, Nin + 1),
		de_dL (N, Nin + 1),
		de_dot (N, 1)
	{
		s_strat = (*rule) (N, Nin + 1, de_W.raw (), de_dL.raw ());
	}

	virtual ~dense_t (void)
	{
	}

	void _sAPI_init (int Nout)
	{
		de_dL.zero ();

		// Glorot, W ~ [-r, r]
		IEEE_t r = sqrt (6.0 / (Nout + s_Nin));
		IEEE_t *p = de_W.raw();
		IEEE_t sample;

		for (int i = de_W.rows () - 1; i >= 0; --i)
			for (int j = de_W.columns () - 1; j >= 0; --j)
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

	virtual void StrategyMono (const int index)
	{
		return; // over-ride when debugging or instrumenting
	}
};

void
dense_t::_sAPI_gradient (stratum_t &Z)
{
	Z.s_delta.TransposeMatrixVectorMult (de_W, s_delta.raw ());
}

void 
dense_t::_sAPI_bprop (IEEE_t *xi, bool activation)
{
	/*
	 *
	 * We are called with,
	 *
	 *    ∂L
	 *  ∑ -- ,
	 *    ∂z
	 *
	 * in s_delta.
	 *
	 */

	// Compute per node delta; skip if activation is the identity
	for (int i = (activation ? 0 : s_Nnodes); i < s_Nnodes; ++i)
		s_delta.sm_data[i] *= DERIVATIVE_FN (s_response.sm_data[i]);

	// Apply the delta for per weight derivatives

	IEEE_t *dL = de_dL.sm_data;
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
dense_t::_sAPI_f (IEEE_t * const xi, bool activate)
{
	de_dot.MatrixVectorMult (de_W, xi);

	IEEE_t *p = s_response.sm_data;
	IEEE_t *dot = de_dot.sm_data;

	if (activate)
		for (int i = 0; i < s_Nnodes; ++i)
			*p++ = ACTIVATION_FN (*dot++);
	else
		for (int i = 0; i < s_Nnodes; ++i)
			*p++ = *dot++;

	return s_response.sm_data;
}

#endif // header inclusion

