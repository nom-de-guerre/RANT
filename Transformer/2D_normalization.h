/*

Copyright (c) 2025, Douglas Santry
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

#ifndef __RANT_MATRIX_NORMALIZATION__H__
#define __RANT_MATRIX_NORMALIZATION__H__

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include <transformer_common.h>
#include <Optimizer.h>

/*
 * Implements layer normalization, Ba et al. 2016
 *
 */

#define GAMMA	0
#define	BETA	1

struct VectorNormalization_t
{
	Md_t				ln_params;
	Md_t				ln_dL;

	IEEE_t				*ln_homep; // points to external memory
	int					ln_N;

	IEEE_t				*ln_xhat;
	IEEE_t				ln_mu;
	IEEE_t				ln_sigma;

	Optimizer_t			ln_O;

	VectorNormalization_t (const int N) :
		ln_params (N, 2),
		ln_dL (ln_params.N (), 2),
		ln_homep (NULL),
		ln_N (-1),
		ln_xhat (new IEEE_t [N]),
		ln_mu (nan ("")),
		ln_sigma (nan ("")),
		ln_O (N, 2, ln_params.raw (), ln_dL.raw ())
	{
		InitLearnable (ln_params.N (), N, ln_params.raw ());
		ln_dL.zero ();
	}

	~VectorNormalization_t (void)
	{
		delete [] ln_xhat;
	}

	void call (const int, IEEE_t * const, IEEE_t *);
	void backward (IEEE_t *, IEEE_t *, IEEE_t *);
	void bprop (IEEE_t *, IEEE_t *);
	void ComputeGout (IEEE_t *, IEEE_t *);

	void update (void)
	{
		ln_O.update ();
	}
};

void
VectorNormalization_t::call (const int N, IEEE_t * const xi, IEEE_t *homep)
{
	ln_homep = homep;
	ln_N = N;

	IEEE_t xx = ln_mu = ln_sigma = 0.0;
	IEEE_t *p = ln_homep;
	IEEE_t *q = ln_params.raw ();

	for (int i = 0; i < N; ++i)
	{
		ln_mu += xi[i];
		xx += xi[i] * xi[i];
	}

	ln_mu /= N;
	ln_sigma = xx / N - ln_mu * ln_mu;
	if (isnan (ln_sigma) || ln_sigma == 0.0)
		ln_sigma = 1;
	else
		ln_sigma = sqrt (ln_sigma);

	if (ln_sigma == 0.0 || isnan (ln_sigma))
		ln_sigma = 1.0;

	for (int i = 0; i < N; ++i, q += 2)
	{
		ln_xhat[i] = p[i] = (xi[i] - ln_mu) / ln_sigma;
		p[i] = q[GAMMA] * p[i] + q[BETA];
	}
}


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
VectorNormalization_t::ComputeGout (IEEE_t *xi, IEEE_t *gradp)
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
	IEEE_t const * const delta = gradp; // s_delta.raw ();

	IEEE_t d_var = 0;
	IEEE_t var_factor = -0.5 * pow (ln_sigma * ln_sigma, -1.5);

	// Eq. 7.24
	for (int i = 0; i < ln_N; ++i)
		d_var += delta[i] * (xi[i] - ln_mu) * var_factor;

	IEEE_t d_mu = 0;
	IEEE_t mu_factor = 0;

	// Eq. 7.25
	for (int i = 0; i < ln_N; ++i)
	{
		d_mu += delta[i] * -1 * 1 / ln_sigma;
		mu_factor += xi[i] - ln_mu;
	}

	mu_factor *= -2;
	mu_factor /= ln_N;

	d_mu += d_var * mu_factor;

	// Eq. 7.26
	for (int i = 0; i < ln_N; ++i)
	{
		gradp[i] = delta[i] * 1 / ln_sigma;
		gradp[i] += d_var * 2 * (xi[i] - ln_mu) / ln_N;
		gradp[i] +=  d_mu / ln_N;
	}
}

void 
VectorNormalization_t::bprop (IEEE_t *delta, IEEE_t *Gout)
{
	IEEE_t *dL = ln_dL.raw ();
	IEEE_t *q = ln_params.raw ();

	for (int i = 0; i < ln_N; ++i, dL += 2, q += 2)
	{
		// Eq. 7.27
		dL[GAMMA] += delta[i] * ln_xhat[i];
		// Eq. 7.28
		dL[BETA] += delta[i];
		// Eq. 7.23
		Gout[i] = delta[i] * q[GAMMA]; // ln_params (i, GAMMA);
	}
}

void
VectorNormalization_t::backward (IEEE_t *xi, IEEE_t *Gin, IEEE_t *Gout)
{
	bprop (Gin, Gout);
	ComputeGout (xi, Gout);
}

class MatrixNormalization_t
{
	int						ma_d;
	VectorNormalization_t	**ma_positions;

	Md_t					ma_X;
	Md_t					ma_Z;
	Md_t					ma_dZ;

public:

	MatrixNormalization_t (const int l, const int d) :
		ma_d (d)
	{
		ma_positions = new VectorNormalization_t * [d];

		for (int i = 0; i < d; ++i)
			ma_positions[i] = new VectorNormalization_t (l);
	}

	~MatrixNormalization_t (void)
	{
		for (int i = 0; i < ma_d; ++i)
			delete ma_positions[i];

		delete [] ma_positions;
	}

	Md_t &call (Md_t &X)
	{
		ma_X = X;

		int rows = X.rows ();
		int Xstride = X.stride ();
		IEEE_t *from = X.raw ();

		ma_Z = Md_t (X.rows (), X.columns ());
		int Zstride = ma_Z.stride ();
		IEEE_t *to = ma_Z.raw ();

		for (int i = 0; i < ma_d; ++i)
		{
			ma_positions[i]->call (rows, from, to);

			from += Xstride;
			to += Zstride;
		}

		return ma_Z;
	}

	Md_t &backward (Md_t &dZ)
	{
		IEEE_t *Gin = dZ.raw ();
		int Gstride = dZ.stride ();

		int Xstride = ma_X.stride ();
		IEEE_t *x = ma_X.raw ();

		ma_dZ = Md_t (ma_X.rows (), ma_X.columns ());
		int outStride = ma_dZ.stride ();
		IEEE_t *Gout = ma_dZ.raw ();

		for (int i = 0; i < ma_d; ++i)
		{
			ma_positions[i]->backward (x, Gin, Gout);

			x += Xstride;
			Gin += Gstride;
			Gout += outStride;
		}

		return ma_dZ;
	}

	void update (void)
	{
		for (int i = 0; i < ma_d; ++i)
			ma_positions[i]->update ();
	}
};

#endif // header inclusion

