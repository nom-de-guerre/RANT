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

#ifndef _NNm_LAYER_NORM__H__
#define _NNm_LAYER_NORM__H__

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include <NeuralM.h>
#include <NNm.h>

/*
 * Implements layer normalization, Ba et al. 2016
 *
 */

#define GAMMA	0
#define	BETA	1

struct layerN_t : public stratum_t
{
	NeuralM_t			ln_params;
	NeuralM_t			ln_dL;
	IEEE_t				*ln_xhat;		// s_response = gamma x xhat + beta
	IEEE_t				ln_mu;
	IEEE_t				ln_sigma;

	layerN_t (const int ID, const int N, StrategyAlloc_t rule) :
		stratum_t ("norm", ID, N, N),
		ln_params (N, 2),
		ln_dL (N, 2),
		ln_xhat (new IEEE_t [N]),
		ln_mu (nan (NULL)),
		ln_sigma (nan (NULL))
	{
		s_strat = (*rule) (N, 2, ln_params.raw (), ln_dL.raw ());
	}

	virtual ~layerN_t (void)
	{
		delete [] ln_xhat;
	}

	void _sAPI_init (void)
	{
		int Nout = s_Nin;

		ln_dL.zero ();

		// Glorot, W ~ [-r, r]
		IEEE_t r = sqrt (6.0 / (Nout + s_Nin));
		IEEE_t *p = ln_params.raw();
		IEEE_t sample;

		for (int i = ln_params.rows () - 1; i >= 0; --i)
			for (int j = ln_params.columns () - 1; j >= 0; --j)
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

/*
 *
 * Compute the gradient to be propagated.
 *
 *        ∂L
 *  𝛿 = ∑ -- , but it is not delta.  It is the first part of its computation.
 *        ∂z
 *
 *
 */

void
layerN_t::_sAPI_gradient (stratum_t &Z)
{
	/*
	 *
	 * We are called with,
	 *
	 *    ∂L
	 *  ∑ -- ,
	 *    ∂z'
	 *
	 * in s_delta.  It was computed in _sAPI_bprop
	 *
	 */

	IEEE_t *gradp = Z.s_delta.raw ();
	IEEE_t *xi = Z.s_response.raw ();

	IEEE_t *delta = s_delta.raw ();

	IEEE_t d_var = 0;
	IEEE_t var_factor = -0.5 * pow (ln_sigma * ln_sigma, -1.5);

	for (int i = 0; i < s_Nnodes; ++i)
		d_var += delta[i] * (xi[i] - ln_mu) * var_factor;

	IEEE_t d_mu = 0;
	IEEE_t mu_factor = 0;

	for (int i = 0; i < s_Nnodes; ++i)
	{
		d_mu += delta[i] * -1 * 1 / ln_sigma;
		mu_factor += xi[i] - ln_mu;
	}

	mu_factor *= -2;
	mu_factor /= s_Nnodes;

	d_mu += d_var * mu_factor;

	for (int i = 0; i < s_Nnodes; ++i)
	{
		gradp[i] = delta[i] * 1 / ln_sigma;
		gradp[i] += d_var * 2 * (xi[i] - ln_mu) / s_Nnodes;
		gradp[i] +=  d_mu / s_Nnodes;
	}
}

void 
layerN_t::_sAPI_bprop (IEEE_t *xi, bool activation)
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

	IEEE_t *dL = ln_dL.raw ();
	IEEE_t *delta = s_delta.raw ();

	for (int i = 0; i < s_Nnodes; ++i, dL += 2)
	{
		dL[GAMMA] += delta[i] * ln_xhat[i];
		dL[BETA] += delta[i];
		delta[i] *= ln_params (i, GAMMA);
	}
}

IEEE_t *
layerN_t::_sAPI_f (IEEE_t * const xi, bool activate)
{
	IEEE_t xx = ln_mu = ln_sigma = 0.0;
	IEEE_t *p = s_response.sm_data;
	IEEE_t *q = ln_params.sm_data;

	for (int i = 0; i < s_Nnodes; ++i)
	{
		ln_mu += xi[i];
		xx += xi[i] * xi[i];
	}

	ln_mu /= s_Nnodes;
	ln_sigma = xx / s_Nnodes - ln_mu * ln_mu;
	if (isnan (ln_sigma) || ln_sigma == 0.0)
		ln_sigma = 1;
	else
		ln_sigma = sqrt (ln_sigma);

	for (int i = 0; i < s_Nnodes; ++i, q += 2)
	{
		ln_xhat[i] = p[i] = (xi[i] - ln_mu) / ln_sigma;
		p[i] = q[GAMMA] * p[i] + q[BETA];
	}

	return s_response.sm_data;
}

#endif // header inclusion

