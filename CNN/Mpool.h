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

#ifndef __DJS_MAXPOOL__H__
#define __DJS_MAXPOOL__H__

#include <layer.h>

class Mpool_t : public mapAPI_t
{
	int				mp_fwidth;		// only square filters currently supported
	plane_t			mp_grad;		// gradient
	int				*mp_rindex;		// reverse index, source of max

public:

	/*
	 * We need the pool width and the input map width.
	 *
	 */
	Mpool_t (const int fwidth, const int iwidth) : 
		mapAPI_t (iwidth / fwidth, iwidth),
		mp_fwidth (fwidth),
		mp_grad (iwidth, iwidth),
		mp_rindex (new int [ma_map.N ()])
	{
		assert ((iwidth % fwidth) == 0);
	}

	~Mpool_t (void)
	{
		delete [] mp_rindex;
	}

	/*
	 * mapAPI_t interface
	 *
	 */
	bool Forward (arg_t &arg)
	{
		assert (arg.a_N == 1);

		return Pool (arg.a_args[0]);
	}

	bool Train (arg_t &arg, double answer)
	{
		mp_grad.Reset ();

		return Forward (arg);
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

bool Mpool_t::Pool (plane_t const * const datap)
{
	__restrict double *omap = ma_map.raw ();
	__restrict int *rindexp = mp_rindex;
	__restrict double *imagep = datap->raw ();

	int idim = datap->rows (); // input image
	int mdim = ma_map.rows (); // output map
	int stride = idim - mp_fwidth;

	for (int start = 0, index = 0, i = 0;
		i < mdim;
		++i, start = mp_fwidth * i * idim)

		for (int i_idx = 0, j = 0; j < mdim; ++j, ++index, start += mp_fwidth)
		{
			omap[index] = -DBL_MAX;
			i_idx = start;

			for (int frow = 0; frow < mp_fwidth; ++frow, i_idx += stride)
				for (int fcol = 0; fcol < mp_fwidth; ++fcol, ++i_idx)
					if (omap[index] < imagep[i_idx])
					{
						omap[index] = imagep[i_idx];
						rindexp[index] = i_idx;
					}
		}

	return true;
}

bool Mpool_t::ComputeGradient (plane_t const * const datap)
{
	__restrict double *gradp = mp_grad.raw ();
	__restrict int *rindexp = mp_rindex;
	__restrict double *deltap = datap->raw ();
	int halt = ma_map.N ();

	for (int i = 0; i < halt; ++i)
		gradp[rindexp[i]] += deltap[i];

	return true;
}

#endif // header inclusion

