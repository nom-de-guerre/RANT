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
	plane_t			**ff_input;
	plane_t			*ff_gradient;
	int				ff_Xi_index;

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
		ff_gradient (NULL),
		ff_Xi_index (-1)
	{
		ff_filter->init (mwidth * mwidth); // on that order, used for Glorot

#ifdef MAX_DEBUG
		ff_filter->s_W.display ("W");
#endif
	}

	filter_t (const int fwidth, const int mwidth, bool delay) :
		mapAPI_t (mwidth - fwidth + 1),
		ff_width (fwidth),
		ff_filter (NULL),
		ff_input (NULL),
		ff_gradient (new plane_t (mwidth, mwidth)),
		ff_Xi_index (-1)
	{
		// work done in setup, this is a striped filter
		ma_cache = mwidth;
	}

	~filter_t (void)
	{
		delete ff_filter;
	}

	void Setup (void)
	{
		ff_filter = new stratum_t (1, ma_stripeN * ff_width * ff_width);
		ff_filter->init (ma_cache * ma_cache);

		ff_input = new plane_t * [ma_stripeN];

		ff_gradient = new plane_t (ma_cache, ma_cache);
	}

	/*
	 * mapAPI_t API implementation.
	 *
	 */
	bool Forward (arg_t &arg)
	{
		if (ma_stripeN < 0)
		{
			ff_input[0] = arg.a_args[0];

			return Convolve (arg.a_args[0]);
		}

		assert (arg.a_N == ma_stripeN);

		double N = ma_map.N ();
		double * __restrict datap = ma_map.raw ();

		int W_stride = ff_width * ff_width;
		double *pW = ff_filter->s_W.raw ();
		double bias = pW[0];
		++pW;

		for (int i = 0; i < N; ++i)
			datap[i] = bias;

		for (int i = 0; i < ma_stripeN; ++i, pW += W_stride)
		{
			ff_input[i] = arg.a_args[i];
			ConvolveStriped (arg.a_args[i], pW);
		}

		return true;
	}

	bool Train (arg_t &arg, double answer)
	{
		ff_Xi_index = 0;
		return Forward (arg);
	}

	bool Backward (arg_t &arg)
	{
		ComputeDerivatives (arg.a_args[0]);
		// we need a copy of the gradient for striped layers
		memcpy (ma_map.raw (), 
			arg.a_args[0]->raw (), 
			ma_map.N () * sizeof (double));

		return true;
	}

	bool Update (void)
	{
		ff_filter->RPROP ();

		return true;
	}

	plane_t *fetchGradient (void)
	{
		ComputeGradient (ff_filter->s_W.raw () + 
			ff_Xi_index + ff_width * ff_width + 
			1);
		++ff_Xi_index;

		return ff_gradient;
	}

	/*
	 * class specific helper functions.
	 *
	 */
	bool ConvolveStriped (plane_t const * const, double const * const);
	bool Convolve (plane_t const * const datap);
	bool ComputeDerivativesStriped (plane_t const * const,
		double *,
		int);
	bool ComputeDerivatives (plane_t const * const datap);
	void ComputeGradient (double const * const);
};

bool filter_t::ConvolveStriped (
	plane_t const * const datap, 
	double const * const pW)
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

#ifdef MAX_DEBUG
	ma_map.display ("filter");
#endif

	return true;
}

bool filter_t::Convolve (plane_t const * const datap)
{
	__restrict double * filterp = ff_filter->s_W.raw ();
	__restrict double * imagep = datap->raw ();
	__restrict double *omap = ma_map.raw ();
	int idim = datap->rows ();
	int stride = idim - ff_width;
	int mdim = ma_map.rows ();

	for (int start = 0, index = 0, i = 0; i < mdim; ++i, start = i * idim)
		for (int i_idx = 0, j = 0; j < mdim; ++j, ++index, ++start)
		{
			omap[index] = filterp[0];		// the bias
			i_idx = start;

			for (int f_idx = 1, k = 0; k < ff_width; ++k, i_idx += stride)
				for (int l = 0; l < ff_width; ++l, ++f_idx, ++i_idx)
					omap[index] += filterp[f_idx] * imagep[i_idx];
		}

#ifdef MAX_DEBUG
	ma_map.display ("filter");
#endif

	return true;
}

void filter_t::ComputeGradient (double const * const pW)
{
	__restrict double const * const filterp = pW; // ff_filter->s_W.raw ();
	__restrict double * imagep = ff_gradient->raw ();
	__restrict double *omap = ma_map.raw (); // contains gradient from next
	int idim = ff_input[0]->rows ();
	int stride = idim - ff_width;
	int mdim = ma_map.rows ();

	for (int start = 0, index = 0, i = 0; i < mdim; ++i, start = i * idim)
		for (int i_idx = 0, j = 0; j < mdim; ++j, ++index, ++start)
		{
			i_idx = start;
			imagep[i_idx] = 0; // no bias contribution

			/*
			 * No bias is passed in, so f_idx starts at 0
			 *
			 */
			for (int f_idx = 0, k = 0; k < ff_width; ++k, i_idx += stride)
				for (int l = 0; l < ff_width; ++l, ++f_idx, ++i_idx)
					imagep[i_idx] += omap[index] * filterp[f_idx];
		}

#ifdef MAX_DEBUG
	ma_map.display ("filter");
#endif
}

bool filter_t::ComputeDerivatives (plane_t const * const datap)
{
	__restrict double * filterp = ff_filter->s_dL.raw ();
	__restrict double * gradientp = datap->raw ();
	__restrict double *inputp = ff_input[0]->raw ();
	int idim = datap->rows ();
	int stride = idim - ff_width;
	int mdim = ma_map.rows ();
	for (int start = 0, index = 0, i = 0; i < mdim; ++i, start = i * idim)
		for (int i_idx = 0, j = 0; j < mdim; ++j, ++index, ++start)
		{
			filterp[0] += gradientp[index];		// the bias
			i_idx = start;

			for (int f_idx = 1, k = 0; k < ff_width; ++k, i_idx += stride)
				for (int l = 0; l < ff_width; ++l, ++f_idx, ++i_idx)
					filterp[f_idx] += inputp[i_idx] * gradientp[index];
		}

	return true;
}

#endif // header inclusion

