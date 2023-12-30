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

/*
 * Implements a layer when training a neural network.
 *
 */
struct dense_t : public stratum_t
{
	// Matrices - per weight, s_Nnodes x s_Nin
	NeuralM_t				de_W;
	NeuralM_t				de_dL;

	dense_t (const int ID, const int N, const int Nin, StrategyAlloc_t rule) :
		stratum_t ("dense", ID, N, Nin + 1),	// account for bias
		de_W (N, Nin + 1),
		de_dL (N, Nin + 1)
	{
		s_strat = (*rule) (N, Nin + 1, de_W.raw (), de_dL.raw ());
	}

	dense_t (const int N, const int Nin, StrategyAlloc_t rule) :
		stratum_t ("dense", -1, N, Nin + 1),	// account for bias
		de_W (N, Nin + 1),
		de_dL (N, Nin + 1)
	{
		s_strat = (*rule) (N, Nin + 1, de_W.raw (), de_dL.raw ());
	}

	dense_t (FILE *fp) : stratum_t ("Dense")
	{
		Load (fp);
	}

	virtual ~dense_t (void)
	{
	}

	void _sAPI_init (void)
	{
		InitLearnable (de_W.N (), de_W.raw ());

		de_dL.zero ();
	}

	virtual IEEE_t * _sAPI_f (IEEE_t * const, bool = true);
	virtual void _sAPI_gradient (stratum_t &);
	virtual void _sAPI_bprop (IEEE_t *, bool = true);

	virtual int _sAPI_Trainable (void)
	{
		return de_W.N ();
	}

	virtual void StrategyMono (const int index)
	{
		return; // over-ride when debugging or instrumenting
	}

	virtual int _sAPI_Store (FILE *fp)
	{
		int bytes;

		bytes = fprintf (fp, "@Dense\n");
		if (bytes < 1)
			return errno;

		Display (NULL, fp);

		bytes = fprintf (fp, "@Dim\t%d\t%d\n", de_W.rows (), de_W.columns ());
		if (bytes < 1)
			return errno;

		de_W.displayExp ("@Weights", fp);

		return 0;
	}

	int Load (FILE *fp)
	{
		char buffer[MAXLAYERNAME];
		int rows, columns;
		int rc;

		shape_t::Load (fp);

		rc = fscanf (fp, "%s %d %d\n", buffer, &rows, &columns);
		if (rc != 3)
			throw ("Invalid dense_t dim");

		if (strcmp ("@Dim", buffer) != 0)
			throw ("dense_t missing @Dim");

		if (rows < 1)
			throw ("invalid dense rows");

		if (columns < 1)
			throw ("invalid dense columns");

		s_Nnodes = rows;
		s_Nin = columns - 1;
		s_response.resize (rows, 1);

		rc = fscanf (fp, "%s\n", buffer);
		if (rc != 1)
			throw ("No Weights");

		rc = de_W.Load (fp, rows, columns);
		if (rc)
			throw ("Bad Dense Weights");

		return 0;
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

	IEEE_t * __restrict dL = de_dL.sm_data;
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
	s_response.MatrixVectorNeuralMult (de_W, xi);

	IEEE_t * __restrict p = s_response.raw ();

	for (int i = 0; i < s_Nnodes; ++i)
	{
		*p = ACTIVATION_FN (*p);
		++p;
	}

	return s_response.raw ();
}

#endif // header inclusion

