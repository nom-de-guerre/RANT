/*

Copyright (c) 2023, Douglas Santry
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

#ifndef __NN_FILTER_MATRIX__H__
#define __NN_FILTER_MATRIX__H__

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
struct filterM_t : public stratum_t
{
	int						cf_k;
	int						cf_stride;

	NeuralM_t				cf_W;	// The filter weights N x k x k
	NeuralM_t				cf_dL;	// The per filter losses

	NeuralM_t				cf_CNN; 	// the kernels
	NeuralM_t				cf_BPROP;
	shape_t					cf_input;	// input shape

	filterM_t (const int ID,
			   const int N,
			   const int k,
			   const int stride,
			   const shape_t &Xin,
			   StrategyAlloc_t rule) :
		stratum_t ("filter2D",
					ID, 
					shape_t (N, 
						1 + (Xin.sh_rows - k) / stride, 
						1 + (Xin.sh_columns - k) / stride)),
		cf_k (k),
		cf_stride (stride),
		cf_W (sh_N, k * k),
		cf_dL (sh_N, k * k),
		cf_CNN (len (), k * k),
		cf_BPROP (N * Xin.mapSize (), k * k), // N is assumed to be sane
		cf_input (Xin)
	{
		s_strat = (*rule) (sh_N, k * k, cf_W.raw (), cf_dL.raw ());
	}

	filterM_t (FILE *fp) : stratum_t ("filter2D")
	{
		Load (fp);
	}

	virtual ~filterM_t (void)
	{
	}

	void _sAPI_init (void)
	{
		InitLearnable (cf_W.N (), cf_W.raw ());

		cf_dL.zero ();
	}

	virtual IEEE_t * _sAPI_f (IEEE_t * const, bool = true);
	virtual void _sAPI_gradient (stratum_t &);
	virtual void _sAPI_bprop (IEEE_t *, bool = true);

	virtual int _sAPI_Trainable (void)
	{
		return cf_W.N ();
	}

	virtual void StrategyMono (const int index)
	{
		return; // over-ride when debugging or instrumenting
	}

	virtual int _sAPI_Store (FILE *fp)
	{
		int bytes;

		bytes = fprintf (fp, "@Filter2D\n");
		if (bytes < 1)
			return errno;

		bytes = fprintf (fp, "@Params\t%d\t%d\n", cf_k, cf_stride);
		if (bytes < 1)
			return errno;

		Display (NULL, fp);
		cf_input.Display ("@Input", fp);

		cf_W.displayMeta (NULL, fp);
		cf_W.displayExp ("@Weights", fp);

		return 0;
	}

	int Load (FILE *fp)
	{
		char buffer[MAXLAYERNAME];
		int rows, columns;
		int rc;

		rc = fscanf (fp, "%s %d %d\n", buffer, &cf_k, &cf_stride);
		if (rc != 3)
			return errno;

		if (strcmp (buffer, "@Params") != 0)
			throw ("Invalid filter2D params");

		shape_t::Load (fp);
		cf_input.Load (fp);

		s_Nnodes = sh_N;
		s_response.resize (sh_N * sh_rows, sh_columns);
		cf_CNN.resize (len (), cf_k * cf_k),

		rc = fscanf (fp, "%s %d, %d\n", buffer, &rows, &columns);
        if (rc != 3)
            throw ("Invalid filter_t dim");

		if (strcmp ("@Meta", buffer) != 0)
			throw ("bad filter2D layer");

		if (rows < 1)
			throw ("filter2D: invalid rows");

		if (columns < 1)
			throw ("filter2D: invalid columns");

		rc = fscanf (fp, "%s\n", buffer);
		if (rc != 1)
			throw ("No Weights");

		cf_W.Load (fp, rows, columns);

		return 0;
	}

	/*
	 * There are per width routines as the loops are unrolled.  This
	 * gives the compiler every chance to optimize.
	 *
	 */
	void BuildConvolutions_2 (IEEE_t const * const inputp);
	void BuildConvolutions_3 (IEEE_t const * const inputp);
	void BuildConvolutions_5 (IEEE_t const * const inputp);

	void BuildDeltaMatrices (void);
	void BPROP_2 (IEEE_t const * const dZ, IEEE_t *M_delta);
	void BPROP_3 (IEEE_t const * const dZ, IEEE_t *M_delta);
	void BPROP_5 (IEEE_t const * const dZ, IEEE_t *M_delta);
};

void
filterM_t::_sAPI_gradient (stratum_t &Z)
{
	BuildDeltaMatrices ();

	Z.s_delta.zero ();

	int iblock = cf_input.mapSize ();
	NeuralM_t dZ (iblock, 1, Z.s_delta.raw ());
	NeuralM_t deltas (cf_input.mapSize (), cf_k * cf_k, cf_BPROP.raw ());

	IEEE_t *f = cf_W.raw ();
	bool skip = cf_input.isSingle ();

	for (int i = 0; i < sh_N; ++i)
	{
		dZ.MatrixVectorMultNoBias (deltas, f);

		deltas.sm_data += deltas.N ();
		f += cf_W.columns ();

		if (skip)
			continue;

		dZ.sm_data += iblock;
	}
}

void 
filterM_t::_sAPI_bprop (IEEE_t *xi, bool activation)
{
	/*
	 *
	 * We are called with,
	 *
	 *    ∂L
	 *  ∑ -- 
	 *    ∂z
	 *
	 * in s_delta.
	 *
	 */

	int beta = sh_rows * sh_columns;
	int feature = cf_k * cf_k;
	int gradIndex = 0;

	bool oneToMany = cf_input.Single ();

	IEEE_t * __restrict gradp = s_delta.raw ();
	IEEE_t const * __restrict CNNp = cf_CNN.raw ();
	IEEE_t const * __restrict F = s_response.raw ();
	IEEE_t * __restrict filter_df = cf_dL.raw ();

	for (int i = 0; i < sh_N; ++i)
	{
		for (int j = 0; j < beta; ++j)
		{
			/*
		 	 *       ∂L
		 	 *  dL = -- 
		 	 *       ∂u
			 *
			 */

			gradp[gradIndex] *= RELU_DERIVATIVE_FN (F[gradIndex]);

			/*
			 *  ∂L   ∂L   ∂u
			 *  -- = -- • -- , the latter term is the input image
			 *  ∂f   ∂u   ∂f
			 *
			 */

			for (int k = 0; k < feature; ++k)
				filter_df[k] += CNNp[k] * gradp[gradIndex];

			CNNp += feature;
			++gradIndex;
		}

		if (oneToMany)
			CNNp = cf_CNN.raw ();

		filter_df += feature;
	}

}

IEEE_t *
filterM_t::_sAPI_f (IEEE_t * const xi, bool activate)
{
	NeuralM_t omap (sh_rows * sh_columns, 1, s_response.raw ());
	IEEE_t *f = cf_W.raw ();
	int oblock = mapSize ();
	int iblock = cf_input.mapSize ();
	IEEE_t const * inputp = xi;
	bool build = true;

	for (int i = 0; i < sh_N; ++i)
	{
		if (build) {
			// 1:1 or 1:many?
			if (cf_input.isSingle ())
				build = false;

			switch (cf_k)
			{
			case 2:

				BuildConvolutions_2 (inputp);
				break;

			case 3:

				BuildConvolutions_3 (inputp);
				break;

			case 5:

				BuildConvolutions_5 (inputp);
				break;

			default:
				assert (false);
			}

			inputp += iblock;
		}

		// Applies RELU activation
		omap.MatrixVectorMultCNN (cf_CNN, f);

		omap.sm_data += oblock;
		f += cf_W.columns ();;
	}

	return s_response.raw ();
}

void 
filterM_t::BuildConvolutions_2 (IEEE_t const * const inputp)
{
	IEEE_t *p0 = cf_CNN.raw ();
	IEEE_t const *im = inputp;
	IEEE_t const * __restrict im0;
	IEEE_t const * __restrict im1;

	for (int i = 0 ; i < sh_rows; ++i)
	{
		im0 = im;
		im1 = im0 + cf_input.sh_columns;

		for (int j = 0; j < sh_columns; ++j) 
		{
			p0[0] = im0[0];
			p0[1] = im0[1];

			p0[2] = im1[0];
			p0[3] = im1[1];

			p0 += 4;

			im0 += cf_stride;
			im1 += cf_stride;
		}

		im += cf_stride * cf_input.sh_columns;
	}
}

void 
filterM_t::BuildConvolutions_3 (IEEE_t const * const inputp)
{
	IEEE_t *p0 = cf_CNN.raw ();
	IEEE_t const *im = inputp;
	IEEE_t const * __restrict im0;
	IEEE_t const * __restrict im1;
	IEEE_t const * __restrict im2;

	for (int i = 0 ; i < sh_rows; ++i)
	{
		im0 = im;
		im1 = im0 + cf_input.sh_columns;
		im2 = im1 + cf_input.sh_columns;

		for (int j = 0; j < sh_columns; ++j) 
		{
			p0[0] = im0[0];
			p0[1] = im0[1];
			p0[2] = im0[2];

			p0[3] = im1[0];
			p0[4] = im1[1];
			p0[5] = im1[2];

			p0[6] = im2[0];
			p0[7] = im2[1];
			p0[8] = im2[2];

			p0 += 9;

			im0 += cf_stride;
			im1 += cf_stride;
			im2 += cf_stride;
		}

		im += cf_stride * cf_input.sh_columns;
	}
}

void 
filterM_t::BuildConvolutions_5 (IEEE_t const * const inputp)
{
	IEEE_t *p0 = cf_CNN.raw ();
	IEEE_t const *im = inputp;
	IEEE_t const * __restrict im0;
	IEEE_t const * __restrict im1;
	IEEE_t const * __restrict im2;
	IEEE_t const * __restrict im3;
	IEEE_t const * __restrict im4;

	for (int i = 0 ; i < sh_rows; ++i)
	{
		im0 = im;
		im1 = im0 + cf_input.sh_columns;
		im2 = im1 + cf_input.sh_columns;
		im3 = im2 + cf_input.sh_columns;
		im4 = im3 + cf_input.sh_columns;

		for (int j = 0; j < sh_columns; ++j) 
		{
			p0[0] = im0[0];
			p0[1] = im0[1];
			p0[2] = im0[2];
			p0[3] = im0[3];
			p0[4] = im0[4];

			p0[5] = im1[0];
			p0[6] = im1[1];
			p0[7] = im1[2];
			p0[8] = im1[3];
			p0[9] = im1[4];

			p0[10] = im2[0];
			p0[11] = im2[1];
			p0[12] = im2[2];
			p0[13] = im2[3];
			p0[14] = im2[4];

			p0[15] = im3[0];
			p0[16] = im3[1];
			p0[17] = im3[2];
			p0[18] = im3[3];
			p0[19] = im3[4];

			p0[20] = im4[0];
			p0[21] = im4[1];
			p0[22] = im4[2];
			p0[23] = im4[3];
			p0[24] = im4[4];

			p0 += 25;

			im0 += cf_stride;
			im1 += cf_stride;
			im2 += cf_stride;
			im3 += cf_stride;
			im4 += cf_stride;
		}

		im += cf_stride * cf_input.sh_columns;
	}
}

void
filterM_t::BuildDeltaMatrices (void)
{
	int Nmaps = N ();
	IEEE_t *deltas = s_delta.raw ();
	IEEE_t *deltaM = cf_BPROP.raw ();

	cf_BPROP.zero ();

	for (int i = 0; i < Nmaps; ++i)
	{
		switch (cf_k)
		{
		case 2:
			BPROP_2 (deltas, deltaM);
			break;
		case 3:
			BPROP_3 (deltas, deltaM);
			break;
		case 5:
			BPROP_5 (deltas, deltaM);
			break;
		default:
			assert (false);
		}

		deltas += mapSize ();
		deltaM += cf_input.mapSize () * cf_k * cf_k;
	}
}

void 
filterM_t::BPROP_2 (IEEE_t const * const deltas, IEEE_t *M)
{
	int features = cf_k * cf_k;
	int jump = features * cf_stride;
	int row_correction = (cf_k - cf_stride) * features +
						(cf_stride - 1) * cf_input.columns () * features;

	IEEE_t correction = 1 + (cf_input.columns () - cf_k) / (IEEE_t) cf_stride;
	IEEE_t dummy;
	correction = modf (correction, &dummy) * cf_stride;
	row_correction += correction * features;

	IEEE_t * __restrict p0 = M;
	IEEE_t * __restrict p1 = p0 + features + 1;
	IEEE_t * __restrict p2 = 
		p1 + features + features * (cf_input.sh_columns - 2) + 1;
	IEEE_t * __restrict p3 = p2 + features + 1;

	IEEE_t const *im = deltas;

	for (int i = 0 ; i < sh_rows; ++i)
	{
		for (int j = 0; j < sh_columns; ++j) 
		{
			*p0 = *p1 = *p2 = *p3 = *im;

			p0 += jump;
			p1 += jump;
			p2 += jump;
			p3 += jump;

			++im;
		}

		p0 += row_correction;
		p1 += row_correction;
		p2 += row_correction;
		p3 += row_correction;
	}
}

void 
filterM_t::BPROP_3 (IEEE_t const * const deltas, IEEE_t *M)
{
	int features = cf_k * cf_k;
	int jump = features * cf_stride;
	int row_correction = (cf_k - cf_stride) * features +
						(cf_stride - 1) * cf_input.columns () * features;

	IEEE_t correction = 1 + (cf_input.columns () - cf_k) / (IEEE_t) cf_stride;
	IEEE_t dummy;
	correction = modf (correction, &dummy) * cf_stride;
	row_correction += correction * features;

	IEEE_t * __restrict p0 = M;
	IEEE_t * __restrict p1 = p0 + features + 1;
	IEEE_t * __restrict p2 = p1 + features + 1;
	IEEE_t * __restrict p3 = 
		p2 + features + features * (cf_input.sh_columns - 3) + 1;
	IEEE_t * __restrict p4 = p3 + features + 1;
	IEEE_t * __restrict p5 = p4 + features + 1;
	IEEE_t * __restrict p6 = 
		p5 + features + features * (cf_input.sh_columns - 3) + 1;
	IEEE_t * __restrict p7 = p6 + features + 1;
	IEEE_t * __restrict p8 = p7 + features + 1;

	IEEE_t const *im = deltas;

	for (int i = 0 ; i < sh_rows; ++i)
	{
		for (int j = 0; j < sh_columns; ++j) 
		{
			*p0 = *p1 = *p2 = *p3 = *p4 = *p5 = *p6 = *p7 = *p8 = *im;

			p0 += jump;
			p1 += jump;
			p2 += jump;
			p3 += jump;
			p4 += jump;
			p5 += jump;
			p6 += jump;
			p7 += jump;
			p8 += jump;

			++im;
		}

		p0 += row_correction;
		p1 += row_correction;
		p2 += row_correction;
		p3 += row_correction;
		p4 += row_correction;
		p5 += row_correction;
		p6 += row_correction;
		p7 += row_correction;
		p8 += row_correction;
	}
}

void 
filterM_t::BPROP_5 (IEEE_t const * const deltas, IEEE_t *M)
{
	int features = cf_k * cf_k;
	int jump = features * cf_stride;
	int row_correction = (cf_k - cf_stride) * features +
						(cf_stride - 1) * cf_input.columns () * features;

	IEEE_t correction = 1 + (cf_input.columns () - cf_k) / (IEEE_t) cf_stride;
	IEEE_t dummy;
	correction = modf (correction, &dummy) * cf_stride;
	row_correction += correction * features;

	IEEE_t * __restrict p0 = M;
	IEEE_t * __restrict p1 = p0 + features + 1;
	IEEE_t * __restrict p2 = p1 + features + 1;
	IEEE_t * __restrict p3 = p2 + features + 1;
	IEEE_t * __restrict p4 = p3 + features + 1;

	IEEE_t * __restrict p5 = 
		p4 + features + features * (cf_input.sh_columns - 5) + 1;
	IEEE_t * __restrict p6 = p5 + features + 1;
	IEEE_t * __restrict p7 = p6 + features + 1;
	IEEE_t * __restrict p8 = p7 + features + 1;
	IEEE_t * __restrict p9 = p8 + features + 1;

	IEEE_t * __restrict p10 = 
		p9 + features + features * (cf_input.sh_columns - 5) + 1;
	IEEE_t * __restrict p11 = p10 + features + 1;
	IEEE_t * __restrict p12 = p11 + features + 1;
	IEEE_t * __restrict p13 = p12 + features + 1;
	IEEE_t * __restrict p14 = p13 + features + 1;

	IEEE_t * __restrict p15 = 
		p14 + features + features * (cf_input.sh_columns - 5) + 1;
	IEEE_t * __restrict p16 = p15 + features + 1;
	IEEE_t * __restrict p17 = p16 + features + 1;
	IEEE_t * __restrict p18 = p17 + features + 1;
	IEEE_t * __restrict p19 = p18 + features + 1;

	IEEE_t * __restrict p20 = 
		p19 + features + features * (cf_input.sh_columns - 5) + 1;
	IEEE_t * __restrict p21 = p20 + features + 1;
	IEEE_t * __restrict p22 = p21 + features + 1;
	IEEE_t * __restrict p23 = p22 + features + 1;
	IEEE_t * __restrict p24 = p23 + features + 1;

	IEEE_t const *im = deltas;

	for (int i = 0 ; i < sh_rows; ++i)
	{
		for (int j = 0; j < sh_columns; ++j) 
		{
			*p0 = *p1 = *p2 = *p3 = *p4 = *p5 = *p6 = *p7 = *p8 =
				*p9 = *p10 = *p11 = *p12 = *p13 = *p14 = *p15 = *p16 = 
				*p17 = *p18 = *p19 = *p20 = *p21 = *p22 = *p23 = *p24 = *im;

			p0 += jump;
			p1 += jump;
			p2 += jump;
			p3 += jump;
			p4 += jump;
			p5 += jump;
			p6 += jump;
			p7 += jump;
			p8 += jump;
			p9 += jump;
			p10 += jump;
			p11 += jump;
			p12 += jump;
			p13 += jump;
			p14 += jump;
			p15 += jump;
			p16 += jump;
			p17 += jump;
			p18 += jump;
			p19 += jump;
			p20 += jump;
			p21 += jump;
			p22 += jump;
			p23 += jump;
			p24 += jump;

			++im;
		}

		p0 += row_correction;
		p1 += row_correction;
		p2 += row_correction;
		p3 += row_correction;
		p4 += row_correction;
		p5 += row_correction;
		p6 += row_correction;
		p7 += row_correction;
		p8 += row_correction;
		p9 += row_correction;
		p10 += row_correction;
		p11 += row_correction;
		p12 += row_correction;
		p13 += row_correction;
		p14 += row_correction;
		p15 += row_correction;
		p16 += row_correction;
		p17 += row_correction;
		p18 += row_correction;
		p19 += row_correction;
		p20 += row_correction;
		p21 += row_correction;
		p22 += row_correction;
		p23 += row_correction;
		p24 += row_correction;
	}
}

#endif // header inclusion

