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

#ifndef __NN_POOLING_MATRIX__H__
#define __NN_POOLING_MATRIX__H__

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include <NeuralM.h>
#include <NNm.h>

/*
 * Implements a pooling layer when training a neural network.
 *
 */
struct poolM_t : public stratum_t
{
	int						cp_k;
	int						cp_stride;
	int						*cp_rindex;	// pooling winners for training

	int						*cp_winners;
	shape_t					cp_input;	// input shape

	poolM_t (const int ID,
			   const int k,
			   const int stride,
			   const shape_t Xin) :
		stratum_t ("pool2D",
					ID, 
					shape_t (Xin.sh_N, 
						1 + (Xin.sh_rows - k) / stride, 
						1 + (Xin.sh_columns - k) / stride)),
		cp_k (k),
		cp_stride (stride),
		cp_winners (new int [len ()]),
		cp_input (Xin)
	{
	}

	poolM_t (FILE *fp) : stratum_t ("pool2D"),
		cp_winners (NULL)
	{
		Load (fp);
	}

	virtual ~poolM_t (void)
	{
		if (cp_winners)
			delete [] cp_winners;
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

	virtual int _sAPI_Store (FILE *fp)
	{
		int bytes;

		bytes = fprintf (fp, "@Maxpool2D\n");
		if (bytes < 1)
			return errno;

		bytes = fprintf (fp, "@Params\t%d\t%d\n", cp_k, cp_stride);
		if (bytes < 3)
			return errno;

		shape_t::Display (NULL, fp);
		cp_input.Display ("@Input", fp);

		return 0;
	}

	int Load (FILE *fp)
	{
		char buffer[MAXLAYERNAME];
		int rc;

		rc = fscanf (fp, "%s %d %d\n", buffer, &cp_k, &cp_stride);
		if (rc < 3)
			throw ("Invalid Pool2D params");

		shape_t::Load (fp);
		cp_input.Load (fp);

		s_Nnodes = sh_N;
		s_response.resize (sh_N * sh_rows, sh_columns);

		cp_winners = new int [len ()];

		return 0;
	}

	void Pool2 (IEEE_t const * const inputp, IEEE_t *, int *);
	void Pool3 (IEEE_t const * const inputp, IEEE_t *, int *);
	void Pool5 (IEEE_t const * const inputp, IEEE_t *, int *);
};

void
poolM_t::_sAPI_gradient (stratum_t &Z)
{
	Z.s_delta.zero ();

	IEEE_t *pZ = Z.s_delta.raw ();
	IEEE_t *dL = s_delta.raw ();

	for (int i = 0; i < sh_length; ++i, ++dL)
		pZ[cp_winners[i]] += *dL;
}

void 
poolM_t::_sAPI_bprop (IEEE_t *xi, bool activation)
{
	// No learnable parameters
}

IEEE_t *
poolM_t::_sAPI_f (IEEE_t * const xi, bool activate)
{
	int oblock = mapSize ();
	int iblock = cp_input.mapSize ();
	IEEE_t const * ip = xi;
	IEEE_t * outp = s_response.raw ();
	int *wp = cp_winners;

	for (int i = 0; i < sh_N; ++i)
	{
		switch (cp_k)
		{
		case 2:
			Pool2 (ip, outp, wp);
			break;

		case 3:
			Pool3 (ip, outp, wp);
			break;

		case 5:
			Pool5 (ip, outp, wp);
			break;

		default:
			assert (false);
		}

		ip += iblock;

		outp += oblock;
		wp += oblock;
	}

	return s_response.raw ();
}

void 
poolM_t::Pool2 (IEEE_t const * const inputp, IEEE_t *outp, int *winners)
{
	IEEE_t const *im = inputp;
	IEEE_t const *im0;
	IEEE_t const *im1;
	int index = 0;
	int base = 0;

	for (int i = 0 ; i < sh_rows; ++i)
	{
		index = base;

		im0 = im;
		im1 = im0 + cp_input.sh_columns;

		for (int j = 0; j < sh_columns; ++j)
		{
			double maximum = im0[0];
			int winner = index;

			if (im0[1] > maximum) {
				maximum = im0[1];
				winner = index + 1;
			}

			if (im1[0] > maximum) {
				maximum = im1[0];
				winner = index + cp_input.sh_columns;
			}

			if (im1[1] > maximum) {
				maximum = im1[1];
				winner = index + cp_input.sh_columns + 1;
			}

			*outp = maximum;
			*winners = winner;

			++outp;
			++winners;

			index += cp_stride;

			im0 += cp_stride;
			im1 += cp_stride;
		}

		im += cp_stride * cp_input.sh_columns;
		base += cp_stride * cp_input.sh_columns;
	}
}

void 
poolM_t::Pool3 (IEEE_t const * const inputp, IEEE_t *outp, int *winners)
{
	IEEE_t const *im = inputp;
	IEEE_t const *im0;
	IEEE_t const *im1;
	IEEE_t const *im2;
	int index = 0;

	for (int i = 0 ; i < sh_rows; ++i)
	{
		im0 = im;
		im1 = im0 + cp_input.sh_columns;
		im2 = im1 + cp_input.sh_columns;

		for (int j = 0; j < sh_columns; ++j)
		{
			double maximum = im0[0];
			int winner = index;

			if (im0[1] > maximum) {
				maximum = im0[1];
				winner = index + 1;
			}

			if (im0[2] > maximum) {
				maximum = im0[2];
				winner = index + 2;
			}

			if (im1[0] > maximum) {
				maximum = im1[0];
				winner = index + cp_input.sh_columns;
			}

			if (im1[1] > maximum) {
				maximum = im1[1];
				winner = index + cp_input.sh_columns + 1;
			}

			if (im1[2] > maximum) {
				maximum = im1[2];
				winner = index + cp_input.sh_columns + 2;
			}

			if (im2[0] > maximum) {
				maximum = im2[0];
				winner = index + 2*cp_input.sh_columns;
			}

			if (im2[1] > maximum) {
				maximum = im2[1];
				winner = index + 2*cp_input.sh_columns + 1;
			}

			if (im2[2] > maximum) {
				maximum = im2[2];
				winner = index + 2*cp_input.sh_columns + 2;
			}

			*outp = maximum;
			*winners = winner;

			++outp;
			++winners;
			index += cp_stride;

			im0 += cp_stride;
			im1 += cp_stride;
			im2 += cp_stride;
		}

		im += cp_input.sh_columns;
	}
}

void 
poolM_t::Pool5 (IEEE_t const * const inputp, IEEE_t *outp, int *winners)
{
	IEEE_t const *im = inputp;
	IEEE_t const *im0;
	IEEE_t const *im1;
	IEEE_t const *im2;
	IEEE_t const *im3;
	IEEE_t const *im4;
	int index = 0;

	for (int i = 0 ; i < sh_rows; ++i)
	{
		im0 = im;
		im1 = im0 + cp_input.sh_columns;
		im2 = im1 + cp_input.sh_columns;
		im3 = im2 + cp_input.sh_columns;
		im4 = im3 + cp_input.sh_columns;

		for (int j = 0; j < sh_columns; ++j)
		{
			double maximum = im0[0];
			int winner = index;

			if (im0[1] > maximum) {
				maximum = im0[1];
				winner = index + 1;
			}

			if (im0[2] > maximum) {
				maximum = im0[2];
				winner = index + 2;
			}

			if (im0[3] > maximum) {
				maximum = im0[3];
				winner = index + 3;
			}

			if (im0[4] > maximum) {
				maximum = im0[4];
				winner = index + 4;
			}

			if (im1[0] > maximum) {
				maximum = im1[0];
				winner = index + cp_input.sh_columns;
			}

			if (im1[1] > maximum) {
				maximum = im1[1];
				winner = index + cp_input.sh_columns + 1;
			}

			if (im1[2] > maximum) {
				maximum = im1[2];
				winner = index + cp_input.sh_columns + 2;
			}

			if (im1[3] > maximum) {
				maximum = im1[3];
				winner = index + cp_input.sh_columns + 3;
			}

			if (im1[4] > maximum) {
				maximum = im1[4];
				winner = index + cp_input.sh_columns + 4;
			}

			if (im2[0] > maximum) {
				maximum = im2[0];
				winner = index + 2*cp_input.sh_columns;
			}

			if (im2[1] > maximum) {
				maximum = im2[1];
				winner = index + 2*cp_input.sh_columns + 1;
			}

			if (im2[2] > maximum) {
				maximum = im2[2];
				winner = index + 2*cp_input.sh_columns + 2;
			}

			if (im2[3] > maximum) {
				maximum = im2[3];
				winner = index + 2*cp_input.sh_columns + 3;
			}

			if (im2[4] > maximum) {
				maximum = im2[4];
				winner = index + 2*cp_input.sh_columns + 4;
			}

			if (im3[0] > maximum) {
				maximum = im3[0];
				winner = index + 3*cp_input.sh_columns;
			}

			if (im3[1] > maximum) {
				maximum = im3[1];
				winner = index + 3*cp_input.sh_columns + 1;
			}

			if (im3[2] > maximum) {
				maximum = im3[2];
				winner = index + 3*cp_input.sh_columns + 2;
			}

			if (im3[3] > maximum) {
				maximum = im3[3];
				winner = index + 3*cp_input.sh_columns + 3;
			}

			if (im3[4] > maximum) {
				maximum = im3[4];
				winner = index + 3*cp_input.sh_columns + 4;
			}

			if (im4[0] > maximum) {
				maximum = im4[0];
				winner = index + 4*cp_input.sh_columns;
			}

			if (im4[1] > maximum) {
				maximum = im4[1];
				winner = index + 4*cp_input.sh_columns + 1;
			}

			if (im4[2] > maximum) {
				maximum = im4[2];
				winner = index + 4*cp_input.sh_columns + 2;
			}

			if (im4[3] > maximum) {
				maximum = im4[3];
				winner = index + 4*cp_input.sh_columns + 3;
			}

			if (im4[4] > maximum) {
				maximum = im4[4];
				winner = index + 4*cp_input.sh_columns + 4;
			}

			*outp = maximum;
			*winners = winner;

			++outp;
			++winners;
			index += cp_stride;

			im0 += cp_stride;
			im1 += cp_stride;
			im2 += cp_stride;
		}

		im += cp_input.sh_columns;
	}
}

#endif // header inclusion

