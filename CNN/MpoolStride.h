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

#ifndef __DJS_MAXPOOLSTRIDE__H__
#define __DJS_MAXPOOLSTRIDE__H__

#include <layer.h>

class MpoolSlide_t : public mapAPI_t
{
	int				mp_fwidth;		// only square filters currently supported
	plane_t			mp_grad;		// gradient
	plane_t			mp_rindex;		// reverse index, source of max

public:

	/*
	 * We need the pool width and the input map width.
	 *
	 */
	MpoolSlide_t (const int fwidth, const int iwidth) : 
		mapAPI_t (iwidth - fwidth + 1),
		mp_fwidth (fwidth),
		mp_grad (iwidth, iwidth),
		mp_rindex (iwidth, iwidth)
	{
	}

	~MpoolSlide_t (void)
	{
	}

	/*
	 * mapAPI_t interface
	 *
	 */
	bool Forward (arg_t &arg)
	{
		return Pool (arg.a_args[0]);
	}

	bool Train (arg_t &arg, double answer)
	{
		int Nentries = mp_grad.N ();
		memset (mp_grad.raw (), 0, Nentries * sizeof (double));

		Pool (arg.a_args[0]);

		return true;
	}

	bool Backward (arg_t &arg)
	{
		assert (arg.a_N == 1);

		ComputeGradient (arg.a_args[0]);

		return true;
	}

	bool Update (void)
	{
		return true;
	}

	plane_t *fetchGradient (void)
	{
		return &mp_grad;
	}

	/*
	 * class specific helper functions.
	 *
	 */
	bool Pool (plane_t const * const datap);
	bool ComputeGradient (plane_t const * const datap);
};

bool MpoolSlide_t::Pool (plane_t const * const datap)
{
	__restrict double *omap = ma_map.raw ();
	__restrict double *rindex = mp_rindex.raw ();
	__restrict double *image = datap->raw ();

	int idim = datap->rows ();
	int mdim = ma_map.rows ();
	int stride = idim - mp_fwidth;

	for (int start = 0, index = 0, i = 0; 
		i < mdim; 
		++i, start = i * idim)
	{
		for (int i_idx = 0, j = 0; 
			j < mdim; 
			++j, ++index, ++start)
		{
			omap[index] = -DBL_MAX;
			i_idx = start;

			for (int frow = 0; frow < mp_fwidth; ++frow, i_idx += stride)
				for (int fcol = 0; fcol < mp_fwidth; ++fcol, ++i_idx)
					if (omap[index] < image[i_idx])
					{
						omap[index] = image[i_idx];
						rindex[index] = i_idx;
					}
		}
	}

	return true;
}

bool MpoolSlide_t::ComputeGradient (plane_t const * const datap)
{
	__restrict double *gradp = mp_grad.raw ();
	__restrict double *rindexp = mp_rindex.raw ();
	__restrict double *deltap = datap->raw ();
	int halt = mp_grad.N ();

	for (int i = 0; i < halt; ++i)
		gradp[(int) rindexp[i]] += deltap[i];

	return true;
}

#endif // header inclusion


