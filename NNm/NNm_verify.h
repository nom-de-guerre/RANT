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

#ifndef __NNM_GRADIENT_VERIFIER__H__
#define __NNM_GRADIENT_VERIFIER__H__

#include <regression.h>

/*
 * This class demonstrates how to use difference equations to verify
 * the implementation of the analytical equations (back propagation).
 *
 * The idea is to use the definition of a derivative.  For a given
 * quantity in the NN:
 *
 *    dL       f (x + h) - f (x - h)
 *    -- = lim --------------------- 
 *    dx   h→0         2h
 *
 * The derivative computed thus should equal the computed BPROP value.
 *
 * A value of h = 1e-7 is a good place to start.
 *
 */

class RegressionGrad_t : public Regression_t
{
	stratum_t			*rg_stratum;

public:

	RegressionGrad_t (int *width, int levels) :
		Regression_t (width, levels, RPROP)
	{
	}

	void VerifyGradient (int level, double h, double *Xi)
	{
		assert (level > -1 && level < n_levels - 1);

		rg_stratum = n_strata[level];

		int N = rg_stratum->N ();

		NeuralM_t G (N, 1, rg_stratum->s_delta.raw ());

		ComputeDerivative (Xi);
		G.Copy ();

		double *ripple = Xi;
		for (int i = 0; i <= level; ++i)
			ripple = n_strata[i]->f (ripple);

		printf ("\tBPROP\t\tDiff\t\tRatio\n");

		// For each of the N perceptrons
		double dL_bp;
		double dL_diff;
		double error;
		double answer = Xi[n_Nin];
		for (int i = 0; i < N; ++i)
		{
			dL_bp = G.raw () [i];

			rg_stratum->s_dot (i, 0) += h;
			rg_stratum->s_response (i, 0) =
				ACTIVATION_FN (rg_stratum->s_dot (i, 0));

			ripple = rg_stratum->s_response.raw ();

			for (int i = level + 1; i < n_levels - 1; ++i)
				ripple = n_strata[i]->f (ripple);

			error = static_cast<Regression_t *>(this)->_API_f (ripple) - answer;
			error = 0.5 * error * error;
			dL_diff = error;

			ripple = rg_stratum->s_response.raw ();

			rg_stratum->s_dot (i, 0) -= 2 * h;
			rg_stratum->s_response (i, 0) =
				ACTIVATION_FN (rg_stratum->s_dot (i, 0));

			for (int i = level + 1; i < n_levels - 1; ++i)
				ripple = n_strata[i]->f (ripple);

			error = static_cast<Regression_t *>(this)->_API_f (ripple) - answer;
			error = 0.5 * error * error;
			dL_diff -= error;
			dL_diff /= 2 * h;

			rg_stratum->s_dot (i, 0) += h;
			rg_stratum->s_response (i, 0) =
				ACTIVATION_FN (rg_stratum->s_dot (i, 0));

			/*
			 *         | bprop | - | diff |
			 * ratio = --------------------
			 *         | bprop | + | diff |
			 *
			 * We're looking for a ratio of zero(ish).
			 *
			 */

			double ratio = (fabs (dL_bp) - fabs (dL_diff)) /
							(fabs (dL_bp) + fabs (dL_diff));

			printf ("∆\t%f\t%f\t%f\n",
				dL_bp,
				dL_diff,
				fabs (ratio));
		}

		UpdateWeights (); // reset the internal state
	}
};

#endif // header inclusion

