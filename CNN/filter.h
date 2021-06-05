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

#ifndef __DJS_FILTER__H__
#define __DJS_FILTER__H__

#include <NNm.h>
#include <layer.h>

class filter_t : public mapAPI_t
{
	int				ff_width;		// only square filters currently supported
	stratum_t		*ff_filter;
	plane_t			**ff_input;		// Xi from the ante layer
	plane_t			const * ff_G;	// gradient from post layer

public:

	/*
	 * We need the filter width and the input map width.
	 *
	 */
	filter_t (const int fwidth, const int mwidth) :
		mapAPI_t (mwidth - fwidth + 1),
		ff_width (fwidth),
		ff_filter (new stratum_t (1, fwidth * fwidth)),
		ff_input (new plane_t *),
		ff_G (NULL)
	{
		ff_filter->init (mwidth * mwidth); // on that order, used for Glorot
	}

	filter_t (const int fwidth, const int mwidth, const int Nin, int *program) :
		mapAPI_t (mwidth - fwidth + 1, Nin, program),
		ff_width (fwidth),
		ff_filter (new stratum_t (1, ma_stripeN * ff_width * ff_width)),
		ff_input (new plane_t * [Nin]),
		ff_G (NULL)
	{
		ff_filter->init (mwidth * mwidth * Nin);
	}

	~filter_t (void)
	{
		delete ff_filter;
		delete [] ff_input;
	}

	/*
	 * mapAPI_t API implementation.
	 *
	 */
	bool Forward (arg_t &arg)
	{
		if (Striped ())
			assert (arg.a_N == ma_stripeN);
		else
			assert (arg.a_N == 1);

		double N = ma_map.N ();
		double * __restrict datap = ma_map.raw ();

		int W_stride = ff_width * ff_width;
		double *pW = ff_filter->s_W.raw ();
		double bias = pW[0];
		++pW;

		for (int i = 0; i < N; ++i)
			datap[i] = bias;

		for (int i = 0; i < arg.a_N; ++i, pW += W_stride)
		{
			ff_input[i] = arg.a_args[i]; // we need it for ∂L/∂f later
			Convolve (arg.a_args[i], pW);
		}

		return true;
	}

	bool Train (arg_t &arg, double answer)
	{
		return Forward (arg);
	}

	bool Backward (arg_t &arg)
	{
		double *bias = ff_filter->s_dL.raw ();
		int blockSize = ma_map.N ();
		__restrict double * gradientp = arg.a_args[0]->raw ();

		ff_G = arg.a_args[0];

		for (int i = 0; i < blockSize; ++i)
			*bias += gradientp[i];

		for (int i = 0; i < arg.a_N; ++i)
			ComputeDerivatives (i);

		return true;
	}

	bool Update (void)
	{
		ff_filter->RPROP ();
		ff_G = NULL;

		return true;
	}

	plane_t *fetchGradient (void)
	{
		assert (false);

		return NULL;
	}

	plane_t *fetchGradient (int index)
	{
		ComputeGradient (index);

		return &ma_map;
	}

	/*
	 * class specific helper functions.
	 *
	 */
	bool ComputeDerivatives (int);
	void ComputeGradient (int);
	bool Convolve (plane_t const * const, double const * const);
};

bool filter_t::Convolve (plane_t const * const datap, double const * const pW)
{
	__restrict double const * const filterp = pW; // ff_filter->s_W.raw ();
	__restrict double * imagep = datap->raw ();
	__restrict double *omap = ma_map.raw ();
	int idim = datap->rows ();
	int stride = idim - ff_width;
	int mdim = ma_map.rows ();

	for (int start = 0, index = 0, i = 0; i < mdim; ++i, start = i * idim)
		for (int i_idx = 0, j = 0; j < mdim; ++j, ++index, ++start)
		{
			i_idx = start;

			/*
			 * No bias is passed in, so f_idx starts at 0
			 *
			 */
			for (int f_idx = 0, k = 0; k < ff_width; ++k, i_idx += stride)
				for (int l = 0; l < ff_width; ++l, ++f_idx, ++i_idx)
					omap[index] += filterp[f_idx] * imagep[i_idx];
		}

	return true;
}

/*
 *              ∂L     ∂L   ∂O  ∂O
 * Compute the: -- = ∑ -- · --, -- = f, transmit the gradient
 *              ∂x     ∂O   ∂x  ∂x
 *
 */

void filter_t::ComputeGradient (int pidx)
{
	// __restrict double const * const filterp = pW; // ff_filter->s_W.raw ();
	int blockSize = ff_width * ff_width;
	__restrict double * filterp = 1 + blockSize * pidx + ff_filter->s_W.raw ();
	__restrict double * i_gradp = ff_G->raw ();
	__restrict double *gradientp = ma_map.raw (); // contains gradient from next
	int idim = ff_input[0]->rows ();
	int stride = idim - ff_width;
	int mdim = ma_map.rows ();

	for (int start = 0, index = 0, i = 0; i < mdim; ++i, start = i * idim)
		for (int i_idx = 0, j = 0; j < mdim; ++j, ++index, ++start)
		{
			i_idx = start;
			gradientp[i_idx] = 0; // no bias contribution

			/*
			 * No bias is passed in, so f_idx starts at 0
			 *
			 */
			for (int f_idx = 0, k = 0; k < ff_width; ++k, i_idx += stride)
				for (int l = 0; l < ff_width; ++l, ++f_idx, ++i_idx)
					gradientp[i_idx] += i_gradp[index] * filterp[f_idx];
			//      ∂O/∂x =             ∑G •             ∂O/∂x 
		}
}

/*
 *              ∂L     ∂L   ∂O  ∂O
 * Compute the: -- = ∑ -- · --, -- = x, used for training the ff_filter.
 *              ∂f     ∂O   ∂f  ∂f
 *
 */

bool filter_t::ComputeDerivatives (int pidx)
{
	int blockSize = ff_width * ff_width;
	__restrict double * filterp = 1 + blockSize * pidx + ff_filter->s_dL.raw ();
	__restrict double * gradientp = ff_G->raw ();
	__restrict double *dO = ff_input[pidx]->raw ();
	int idim = ff_G->rows ();
	int stride = idim - ff_width;
	int mdim = ma_map.rows ();

	for (int start = 0, index = 0, i = 0; i < mdim; ++i, start = i * idim)
		for (int i_idx = 0, j = 0; j < mdim; ++j, ++index, ++start)
		{
			i_idx = start;

			for (int f_idx = 0, k = 0; k < ff_width; ++k, i_idx += stride)
				for (int l = 0; l < ff_width; ++l, ++f_idx, ++i_idx)
					filterp[f_idx] += dO[i_idx] * gradientp[index];
		}

	return true;
}

#endif // header inclusion

