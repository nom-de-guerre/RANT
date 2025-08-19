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

#ifndef _NN_DIFFERENCING__H__
#define _NN_DIFFERENCING__H__

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include <NeuralM.h>
#include <NNm.h>

/*
 * This class demonstrates how to use difference equations to verify
 * the implementation of the analytical equations (back propagation).
 *
 * The idea is to use the definition of a derivative.  For a given
 * quantity in the NN:
 *
 *    dL       L (x + h) - L (x - h)
 *    -- = lim --------------------- 
 *    dx   h→0         2h
 *
 * The derivative computed thus should equal the computed BPROP value.
 *
 * A value of h = 1e-7 is a good place to start.
 *
 */

struct verify_t : public stratum_t
{
	verify_t (const int ID, const int N) :
		stratum_t ("verify", ID, N, N)
	{
	}

	virtual ~verify_t (void)
	{
	}

	void _sAPI_init (void)
	{
	}

	virtual IEEE_t * _sAPI_f (IEEE_t * const, bool = true);
	virtual void _sAPI_gradient (stratum_t &);
	virtual void _sAPI_bprop (IEEE_t *, bool = true);

	virtual int _sAPI_Trainable (void)
	{
		return 0;
	}

	virtual void StrategyMono (const int index)
	{
		return; // over-ride when debugging or instrumenting
	}

	void VerifyGradient (NNet_t *p, IEEE_t h, IEEE_t *Xi, bool verbose=false)
	{
		NeuralM_t G (s_Nnodes, 1, s_delta.raw ());

		int level = s_ID; // Currently a layer's ID is its index in n_strata

		p->ComputeDerivative (Xi);
		G.Copy ();

		IEEE_t *ripple = Xi;
		for (int i = 0; i <= level; ++i)
			ripple = p->n_strata[i]->_sAPI_f (ripple);

		printf ("\tIndex\tBPROP\t\tDiff\t\tRatio\n");

		// For each of the N perceptrons
		IEEE_t dL_bp;
		IEEE_t dL_diff;
		IEEE_t error;
		IEEE_t answer = Xi[p->n_Nin];
		IEEE_t save;
		stratum_t *bottomp = p->Bottom ();

		for (int i = 0; i < s_Nnodes; ++i)
		{
			dL_bp = G (i, 0);

			save = s_response (i, 0);

			s_response (i, 0) += h;

			ripple = s_response.raw ();

			for (int i = level + 1; i < p->n_populated; ++i)
				ripple = p->n_strata[i]->_sAPI_f (ripple);

			dL_diff = bottomp->_sAPI_Loss (&answer);

			ripple = s_response.raw ();

			s_response (i, 0) = save - h;

			for (int i = level + 1; i < p->n_populated; ++i)
				ripple = p->n_strata[i]->_sAPI_f (ripple);

			error = bottomp->_sAPI_Loss (&answer);

			dL_diff -= error;
			dL_diff /= 2 * h;

			s_response (i, 0) = save;

			/*
			 *         | bprop | - | diff |
			 * ratio = --------------------
			 *         | bprop | + | diff |
			 *
			 * We're looking for a ratio of zero(ish).
			 *
			 */

			IEEE_t denom = (fabs (dL_bp) + fabs (dL_diff));
			IEEE_t ratio = fabs (dL_bp) - fabs (dL_diff);
			if (denom > 0)
				ratio /= denom;
			else
				ratio = 0.0;

			if (fabs (ratio) > 1e-4 || verbose)
				printf ("∆\t%d\t%e\t%e\t%f\n",
					i,
					dL_bp,
					dL_diff,
					fabs (ratio));
		}

		p->UpdateWeights (); // reset the internal state
	}

};

void
verify_t::_sAPI_gradient (stratum_t &Z)
{
	Z.s_delta.Accept (s_delta.raw ());
}

void 
verify_t::_sAPI_bprop (IEEE_t *xi, bool activation)
{
}

IEEE_t *
verify_t::_sAPI_f (IEEE_t * const xi, bool activate)
{
	s_response.Accept (xi);

	return xi;
}

#endif // header inclusion

