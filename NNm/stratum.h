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

#ifndef _NN_STRATUM__H__
#define _NN_STRATUM__H__

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include <NeuralM.h>

#define RECTIFIER(X) log (1 + exp (X))
#define SIGMOID_FN(X) (1 / (1 + exp (-X))) // derivative of rectifier

#ifdef __TANH_ACT_FN
#define ACTIVATION_FN(X) tanh(X)
#define DERIVATIVE_FN(Y) (1 - Y*Y)
#else
#define ACTIVATION_FN(X) SIGMOID_FN(X)
#define DERIVATIVE_FN(Y) (Y * (1 - Y))
#endif

#ifndef SIGN
#define SIGN(X) (signbit (X) != 0 ? -1.0 : 1.0)
#endif

/*
 * Implements a layer when training a neural network.
 *
 */
struct stratum_t
{
	int						s_Nperceptrons;
	int						s_Nin; 				// # weights, includes bias

	// Matrices - per weight, s_Nperceptrons x s_Nin
	NeuralM_t				s_W;
	NeuralM_t				s_dL;
	NeuralM_t				s_deltaW;

	// Vectors - per perceptron (node)
	NeuralM_t				s_delta;
	NeuralM_t				s_dot;
	NeuralM_t				s_response;

	stratum_t (const int N, const int Nin) :
		s_Nperceptrons (N),
		s_Nin (Nin + 1),				// account for bias
		s_W (s_Nperceptrons, s_Nin),
		s_dL (s_Nperceptrons, s_Nin),
		s_deltaW (s_Nperceptrons, s_Nin),
		s_delta (s_Nperceptrons, 1),
		s_dot (s_Nperceptrons, 1),
		s_response (s_Nperceptrons, 1)
	{
		s_dL.zero ();
	}

	virtual ~stratum_t (void)
	{
	}

	void init (int Nout)
	{
		// Glorot, W ~ [-r, r]
		double r = sqrt (6.0 / (Nout + s_Nin));
		double *p = s_W.raw();
		double sample;

		for (int i = s_W.rows () - 1; i >= 0; --i)
			for (int j = s_W.columns () - 1; j >= 0; --j)
			{
				sample = (double) rand () / RAND_MAX;
				sample *= r;
				if (rand () % 2)
					sample = -sample;
				*p++ = sample;
			}
	}

	int N (void)
	{
		return s_Nperceptrons;
	}

	void bprop (stratum_t &, double *);
	double *f (double *, bool = true);
	double *f (double *, double *);

	virtual void Strategy (void) = 0;
};

void 
stratum_t::bprop (stratum_t &next, double *xi)
{
	/*
	 * Compute per node total derivative for the layer.
	 *
	 * ∂L           ∂∑
	 * -- = ∑ ( 𝛿 · -- ), the right-hand side referring to the next level
	 * ∂y           ∂x
	 *
	 */
	s_delta.TransposeMatrixVectorMult (next.s_W, next.s_delta.raw ());

	// Compute per node delta
	for (int i = 0; i < s_Nperceptrons; ++i)
		s_delta.sm_data[i] *= DERIVATIVE_FN (s_response.sm_data[i]);

	// Apply the delta for per weight derivatives

	double *dL = s_dL.sm_data;
	double delta;

	/*
	 * ∂L     ∂∑
	 * -- = 𝛿 --
	 * ∂w     ∂w
	 *
	 */
	for (int i = 0; i < s_Nperceptrons; ++i)
	{
		delta = s_delta.sm_data[i];
		*dL++ += delta; // the Bias

		for (int j = 1; j < s_Nin; ++j)
			*dL++ += delta * xi[j - 1];
	}
}

double *
stratum_t::f (double *xi, bool activate)
{
	s_dot.MatrixVectorMult (s_W, xi);

	double *p = s_response.sm_data;
	double *dot = s_dot.sm_data;

	if (activate)
		for (int i = 0; i < s_Nperceptrons; ++i)
			*p++ = ACTIVATION_FN (*dot++);
	else
		for (int i = 0; i < s_Nperceptrons; ++i)
			*p++ = *dot++;

	return s_response.sm_data;
}

double *
stratum_t::f (double *xi, double *result)
{
	s_dot.MatrixVectorMult (s_W, xi);

	double *p = s_response.sm_data;
	double *dot = s_dot.sm_data;
	for (int i = 0; i < s_Nperceptrons; ++i)
		*p++ = result[i] = ACTIVATION_FN (*dot++);

	return s_response.sm_data;
}

#endif // header inclusion

