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
	int				mp_width;		// only square filters currently supported
	plane_t			mp_grad;		// gradient
	plane_t			mp_rindex;		// reverse index, source of max

public:

	/*
	 * We need the pool width and the input map width.
	 *
	 */
	Mpool_t (const int mwidth, const int iwidth) : 
		mapAPI_t (iwidth - mwidth + 1),
		mp_width (mwidth),
		mp_grad (iwidth, iwidth),
		mp_rindex (iwidth, iwidth)
	{
	}

	~Mpool_t (void)
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
		memset (mp_grad.raw (), 0, mp_grad.N () * sizeof (double));
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

bool Mpool_t::Pool (plane_t const * const datap)
{
	__restrict double *omap = ma_map.raw ();
	__restrict double *rindexp = mp_rindex.raw ();
	__restrict double *imagep = datap->raw ();

	int idim = datap->rows ();
	int mdim = ma_map.rows ();
	int stride = idim - mp_width;

	for (int start = 0, index = 0, i = 0; i < mdim; ++i, start = i * idim)
		for (int i_idx = 0, j = 0; j < mdim; ++j, ++index, ++start)
		{
			omap[index] = -DBL_MAX;
			i_idx = start;

			for (int k = 0; k < mp_width; ++k, i_idx += stride)
				for (int l = 0; l < mp_width; ++l, ++i_idx)
					if (omap[index] < imagep[i_idx])
					{
						omap[index] = imagep[i_idx];
						rindexp[index] = i_idx;
					}
		}

#ifdef MAX_DEBUG
	ma_map.display ("Mpool");
	for (int i = 0; i < MapSize () ; ++i)
{
		printf ("%d  ", (int) rindexp[i]);
assert (rindexp[i] < 676);
}
	printf ("\n");
#endif

	return true;
}

bool Mpool_t::ComputeGradient (plane_t const * const datap)
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

