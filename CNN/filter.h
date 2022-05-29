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
	dense_t			*ff_filter;
	plane_t			**ff_input;		// Xi from the ante layer
	plane_t			*ff_flux;		// gradient we propagate backwards
	IEEE_t	const 	*ff_G;	// gradient from post layer

public:

	/*
	 * We need the filter width and the input map width.
	 *
	 */
	filter_t (const int fwidth, const int mwidth) :
		mapAPI_t (mwidth - fwidth + 1, mwidth),
		ff_width (fwidth),
		ff_filter (new dense_t (1, fwidth * fwidth, RPROP)),
		ff_input (new plane_t *),
		ff_flux (new plane_t (mwidth, mwidth)),
		ff_G (NULL)
	{
		// on that order, used for Glorot
		ff_filter->_sAPI_init (mwidth * mwidth);
	}

	filter_t (const int fwidth, const int mwidth, const int Nin, int *program) :
		mapAPI_t (mwidth - fwidth + 1, Nin, program),
		ff_width (fwidth),
		ff_filter (new dense_t (1, ma_stripeN * ff_width * ff_width, RPROP)),
		ff_input (new plane_t * [Nin]),
		ff_flux (new plane_t (mwidth, mwidth)),
		ff_G (NULL)
	{
		ma_iwidth = mwidth;
		ff_filter->_sAPI_init (mwidth * mwidth * Nin);
	}

	~filter_t (void)
	{
		delete ff_filter;
		delete [] ff_input;
		delete ff_flux;
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

		IEEE_t N = ma_map.N ();
		IEEE_t * __restrict datap = ma_map.raw ();

		int W_stride = ff_width * ff_width;
		IEEE_t *pW = ff_filter->de_W.raw ();
		IEEE_t bias = pW[0];
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

	bool Train (arg_t &arg, IEEE_t answer)
	{
		return Forward (arg);
	}

	bool Backward (arg_t &arg)
	{
		IEEE_t *bias = ff_filter->de_dL.raw ();
		int blockSize = ma_map.N ();
		__restrict IEEE_t * gradientp = arg.a_args[0]->raw ();

		ff_G = arg.a_args[0]->raw ();

		assert (arg.a_N == 1);

		for (int i = 0; i < blockSize; ++i)
			*bias += gradientp[i];

		for (int i = 0; i < arg.a_N; ++i)
			ComputeDerivatives (i);

		return true;
	}

	bool Update (void)
	{
		ff_filter->_sAPI_strategy ();
#ifndef __SAVE_G_FOR_VERIFICATION
		ff_G = NULL;
#endif

		return true;
	}

	plane_t *fetchGradient (void)
	{
		return fetchGradient (0);
	}

	plane_t *fetchGradient (int index)
	{
		ff_flux->Reset ();
		ComputeGradient (index);

		return ff_flux;
	}

	/*
	 * class specific helper functions.
	 *
	 */
	bool ComputeDerivatives (int);
	void ComputeGradient (int);
	bool Convolve (plane_t const * const, IEEE_t const * const);
};

bool filter_t::Convolve (plane_t const * const datap, IEEE_t const * const pW)
{
	__restrict IEEE_t const * const filterp = pW;
	__restrict IEEE_t * imagep = datap->raw ();
	__restrict IEEE_t *omap = ma_map.raw ();
	int idim = datap->rows ();
	int stride = idim - ff_width;
	int mdim = ma_map.rows ();

	for (int start = 0, index = 0, i = 0; i < mdim; ++i, start = i * idim)
		for (int i_idx = 0, j = 0; j < mdim; ++j, ++index, ++start)
		{
			i_idx = start;

			// bias set prior to call to Convolve ().
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
	int blockSize = ff_width * ff_width;
	__restrict IEEE_t * filterp = 1 + blockSize * pidx + ff_filter->de_W.raw ();
	__restrict IEEE_t const * i_gradp = ff_G;
	__restrict IEEE_t *gradientp = ff_flux->raw ();
	int idim = inputSize ();
	int stride = idim - ff_width;
	int mdim = ma_map.rows ();

	for (int start = 0, index = 0, i = 0; i < mdim; ++i, start = i * idim)
		for (int i_idx = 0, j = 0; j < mdim; ++j, ++index, ++start)
		{
			i_idx = start;

			for (int f_idx = 0, k = 0; k < ff_width; ++k, i_idx += stride)
				for (int l = 0; l < ff_width; ++l, ++f_idx, ++i_idx)
					gradientp[i_idx] += i_gradp[index] * filterp[f_idx];
			//      ∂L/∂x =          ∑       G (∂L/∂O) • ∂O/∂x
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
	__restrict IEEE_t * dW = 1 + blockSize * pidx + ff_filter->de_dL.raw ();
	__restrict IEEE_t const * dO = ff_G;
	__restrict IEEE_t *input = ff_input[pidx]->raw ();
	IEEE_t *bias = ff_filter->de_dL.raw ();
	int idim = ma_map.rows ();
	int stride = idim - ff_width;
	int mdim = ma_map.rows ();

	for (int start = 0, index = 0, i = 0; i < mdim; ++i, start = i * idim)
		for (int i_idx = 0, j = 0; j < mdim; ++j, ++index, ++start)
		{
			i_idx = start;
			*bias += dO[index];

			for (int f_idx = 0, k = 0; k < ff_width; ++k, i_idx += stride)
				for (int l = 0; l < ff_width; ++l, ++f_idx, ++i_idx)
					dW[f_idx] += dO[index] * input[i_idx];
			//      ∂L/∂f =   ∑   (∂L/∂O)  • ∂O/∂f
		}

	return true;
}

#endif // header inclusion

