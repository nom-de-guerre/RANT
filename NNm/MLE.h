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

#ifndef _NN_SOFTMAX_MLE__H__
#define _NN_SOFTMAX_MLE__H__

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
struct SoftmaxMLE_t : public stratum_t
{
	// Matrices - per weight, s_Nnodes x s_Nin
	NeuralM_t				ml_W;
	NeuralM_t				ml_dL;

	Softmax_t				ml_softm;

	IEEE_t					ml_guess;

	SoftmaxMLE_t (const int ID,
			const int K, 
			const int Nin,
			StrategyAlloc_t rule) :
		stratum_t ("MLE", ID, K, Nin + 1),	// account for bias
		ml_W (K, Nin + 1),
		ml_dL (K, Nin + 1),
		ml_softm (K),
		ml_guess (-1)
	{
		s_strat = (*rule) (K, Nin + 1, ml_W.raw (), ml_dL.raw ());
	}

	SoftmaxMLE_t (FILE *fp, const int K) :
		stratum_t ("MLE"),
		ml_softm (K)
	{
		Load (fp);
	}

	virtual ~SoftmaxMLE_t (void)
	{
	}

	void _sAPI_init (void)
	{
		InitLearnable (ml_W.N (), ml_W.raw ());

		ml_dL.zero ();
	}

	virtual IEEE_t * _sAPI_f (IEEE_t * const, bool = true);
	virtual void _sAPI_gradient (stratum_t &);
	virtual void _sAPI_bprop (IEEE_t *, bool = true);
	virtual IEEE_t _sAPI_Loss (IEEE_t const * const);

	virtual int _sAPI_Trainable (void)
	{
		return ml_W.N ();
	}

	virtual void StrategyMono (const int index)
	{
		return; // over-ride when debugging or instrumenting
	}

	virtual int _sAPI_Store (FILE *fp)
	{
		fprintf (fp, "@MLE\n");
		Display (NULL, fp);
		fprintf (fp, "@Dim\t%d\t%d\n", ml_W.rows (), ml_W.columns ());
		ml_W.displayExp ("@Weights", fp);

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
            throw ("Invalid MLE dim");

        if (strcmp ("@Dim", buffer) != 0)
            throw ("Invalid MLE dim");

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

        rc = ml_W.Load (fp, rows, columns);
        if (rc)
            throw ("Bad MLE Weights");

        return 0;
	}
};

void
SoftmaxMLE_t::_sAPI_gradient (stratum_t &Z)
{
	Z.s_delta.TransposeMatrixVectorMult (ml_W, s_delta.raw ());
}

void 
SoftmaxMLE_t::_sAPI_bprop (IEEE_t *xi, bool activation)
{
	IEEE_t *dL = ml_dL.sm_data;
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
SoftmaxMLE_t::_sAPI_f (IEEE_t * const xi, bool activate)
{
	s_response.MatrixVectorMultBias (ml_W, xi);

	ml_guess = ml_softm.ComputeSoftmax (s_response.raw ());

	return &ml_guess;
}

IEEE_t
SoftmaxMLE_t::_sAPI_Loss (IEEE_t const * const answers)
{
	IEEE_t loss = -log (ml_softm.P (answers[0]));
	if (isnan (loss) || isinf (loss))
		loss = 1000000;

	ml_softm.bprop (answers[0], s_delta.raw ());

	return loss;
}

#endif // header inclusion

