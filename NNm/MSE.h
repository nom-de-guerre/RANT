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
 * Implements the mean squared error (MSE) loss function.
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

	ScalerMSE_t (FILE *fp) : stratum_t ("MSE")
	{
		Load (fp);
	}

	virtual ~ScalerMSE_t (void)
	{
	}

	void _sAPI_init (void)
	{
		InitLearnable (ms_W.N (), ms_W.raw ());

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

	virtual int _sAPI_Store (FILE *fp)
	{
		fprintf (fp, "@MSE\n");
		fprintf (fp, "@Dim\t%d\t%d\n", ms_W.rows (), ms_W.columns ());
		ms_W.displayExp ("@Weights", fp);

		return 0;
	}

	int Load (FILE *fp)
	{
		char buffer[MAXLAYERNAME];
        int rows, columns;
        int rc;

        rc = fscanf (fp, "%s %d %d\n", buffer, &rows, &columns);
        if (rc != 3)
            throw ("Invalid MSE dim");

		if (strcmp ("@Dim", buffer) != 0)
			throw ("Invalid MSE dim");

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

        rc = ms_W.Load (fp, rows, columns);
        if (rc)
            throw ("Bad MSE Weights");

		return 0;
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
	s_response.MatrixVectorNeuralMult (ms_W, xi);

	IEEE_t *f = s_response.raw ();

	if (activate)
		for (int i = 0; i < s_Nin; ++i, ++f)
			*f = ACTIVATION_FN (*f);

	return s_response.raw ();
}

IEEE_t
ScalerMSE_t::_sAPI_Loss (IEEE_t const * const answers)
{
	IEEE_t loss;

	loss = s_delta.sm_data[0] = s_response.sm_data[0] - answers[0];
	loss *= loss;

	return 0.5 * loss;
}

#endif // header inclusion

