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

#include <NNm.h>
#include <convolve.h>

class Mpool_t : public discrete_t
{
	int				mp_fwidth;		// only square filters currently supported
	int				mp_idim;
	int				mp_odim;
	int				*mp_rindex;		// reverse index, source of max

public:

	/*
	 * We need the pool width and the input map width.
	 *
	 */
	Mpool_t (const int fwidth, const int iwidth, StrategyAlloc_t rule) : 
		mp_fwidth (fwidth),
		mp_idim (iwidth),
		mp_odim (iwidth / fwidth),
		mp_rindex (new int [iwidth * iwidth])
	{
	}

	Mpool_t (FILE *fp, const int No, const int fwidth, const int mwidth) :
		mp_rindex (NULL)
	{
		char buffer[32];
		int vNo;
		int rc = fscanf (fp, "%s %d\t%d\t%d\t%d\n",
			buffer,
			&vNo,
			&mp_fwidth,
			&mp_idim,
			&mp_odim);

		assert (rc == 5);
		assert (vNo == No);
		assert (fwidth == mp_fwidth);
		assert (mwidth == mp_odim);
	}

	virtual ~Mpool_t (void)
	{
		delete [] mp_rindex;
	}

	virtual void f (IEEE_t *inputp, IEEE_t *outputp)
	{
		Pool (inputp, outputp);
	}

	virtual void UpdateWeights (void)
	{
		// no trainable parameters
	}

	virtual void BPROP (IEEE_t *fromp, IEEE_t *top)
	{
		// no trainable parameters
	}

	virtual void Propagate (IEEE_t *fromp, IEEE_t *top)
	{
		ComputeGradient (fromp, top);
	}

	/*
	 * class specific helper functions.
	 *
	 */
	void Pool (IEEE_t const * const , IEEE_t * __restrict);
	void ComputeGradient (IEEE_t const * const, IEEE_t *gradp);

	virtual int Persist (FILE *fp, const int No)
	{
		fprintf (fp, "@MaxPool %d\t%d\t%d\t%d\n",
			No,
			mp_fwidth,
			mp_idim,
			mp_odim);

		return 0;
	}
};

void
Mpool_t::Pool (
	IEEE_t const * const imagep,
	IEEE_t * __restrict omap)
{
	int * __restrict rindexp = mp_rindex;
	bool reset = true;

	for (int base = 0, linear = 0, row = 0; row < mp_idim; ++row)
	{
		if (row && (row % mp_fwidth) == 0)
		{
			++base;
			reset = true;
		}

		int index = base * mp_odim;

		if (reset)
			omap[index] = -DBL_MAX;

		for (int column = 0; column < mp_idim; ++column, ++linear)
		{
			if (column && (column % mp_fwidth == 0))
			{
				++index;
				if (reset)
					omap[index] = -DBL_MAX;
			}

			if (omap[index] < imagep[linear])
			{
				omap[index] = imagep[linear];
				if (rindexp)
					rindexp[index] = linear;
			}
		}

		reset = false;
	}
}

void
Mpool_t::ComputeGradient (
	IEEE_t const * const deltap,
	IEEE_t * __restrict gradp)
{
	int * __restrict rindexp = mp_rindex;
	int halt = mp_idim * mp_idim;

	memset (gradp, 0, halt * sizeof (IEEE_t));

	for (int i = 0; i < halt; ++i)
		gradp[rindexp[i]] += deltap[i];
}

#endif // header inclusion


