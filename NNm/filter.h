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
#include <convolve.h>

class filter_t : public discrete_t
{
	int				ff_width;		// only square filters currently supported
	int				ff_isize;
	int				ff_osize;

	NeuralM_t		ff_W;
	NeuralM_t		ff_dL;

	strategy_t		*ff_strategy;

public:

	/*
	 * We need the filter width and the input map width.
	 *
	 */
	filter_t (const int fwidth, const int mwidth, StrategyAlloc_t rule) :
		ff_width (fwidth),
		ff_isize (mwidth),
		ff_osize (mwidth - fwidth + 1),
		ff_W (1, 1 + fwidth * fwidth),
		ff_dL (1, 1 + fwidth * fwidth)
	{
		ff_strategy = (*rule) (1, 1+fwidth * fwidth, ff_W.raw (), ff_dL.raw ());

		for (int i = 0; i < ff_W.N (); ++i)
			ff_W (0, i) = 1.0; // (double) rand () / RAND_MAX;
			// ff_W (0, i) = (double) rand () / RAND_MAX;

		ff_dL.zero ();
	}

	~filter_t (void)
	{
		delete ff_strategy;
	}

	virtual void f (IEEE_t *inputp, IEEE_t *outputp)
	{
		Convolve (inputp, outputp);
	}

	virtual void UpdateWeights (void)
	{
		ff_strategy->_tAPI_strategy ();
	}

	virtual void BPROP (IEEE_t *gradp, IEEE_t *inputp)
	{
		ComputeDerivatives (gradp, inputp);
	}

	virtual void Propagate (IEEE_t *fromp, IEEE_t *top)
	{
		ComputeGradient (fromp, top);
	}

	void ComputeDerivatives (IEEE_t *, IEEE_t *);
	void Convolve (IEEE_t const * const, IEEE_t *);
	void ComputeGradient (IEEE_t *, IEEE_t *);
};

void
filter_t::Convolve (
	IEEE_t const * const imagep,
	IEEE_t * __restrict omap)
{
	IEEE_t const * const filterp = ff_W.raw () + 1;
	int stride = ff_isize - ff_width;

	for (int start = 0, index = 0, i = 0;
		 i < ff_osize;
		 ++i, start = i * ff_isize)
	{
		for (int i_idx = 0, j = 0; j < ff_osize; ++j, ++index, ++start)
		{
			i_idx = start;

			omap[index] = filterp[-1]; // the bias
			for (int f_idx = 0, k = 0; k < ff_width; ++k, i_idx += stride)
				for (int l = 0; l < ff_width; ++l, ++f_idx, ++i_idx)
					omap[index] += filterp[f_idx] * imagep[i_idx];
		}
	}
}

/*
 *              ∂L     ∂L   ∂O  ∂O
 * Compute the: -- = ∑ -- · --, -- = f, transmit the gradient
 *              ∂x     ∂O   ∂x  ∂x
 *
 */
void
filter_t::ComputeGradient (IEEE_t * __restrict fromp, IEEE_t * __restrict top)
{
	IEEE_t * __restrict filterp = 1 + ff_W.raw ();
	int stride = ff_isize - ff_width;

	for (int start = 0, index = 0, i = 0;
		 i < ff_osize;
		 ++i, start = i * ff_isize)
	{
		for (int i_idx = 0, j = 0; j < ff_osize; ++j, ++index, ++start)
		{
			i_idx = start;

			for (int f_idx = 0, k = 0; k < ff_width; ++k, i_idx += stride)
				for (int l = 0; l < ff_width; ++l, ++f_idx, ++i_idx)
					top[i_idx] += fromp[index] * filterp[f_idx];
			//      ∂L/∂x =          ∑       G (∂L/∂O) • ∂O/∂x
		}
	}
}

/*
 *              ∂L     ∂L   ∂O  ∂O
 * Compute the: -- = ∑ -- · --, -- = x, used for training the ff_filter.
 *              ∂f     ∂O   ∂f  ∂f
 *
 */

void
filter_t::ComputeDerivatives (IEEE_t * __restrict dO, IEEE_t * __restrict input)
{
	IEEE_t * __restrict dW = ff_dL.raw () + 1;
	int stride = ff_isize - ff_width;

	for (int start = 0, index = 0, i = 0;
		 i < ff_osize;
		 ++i, start = i * ff_isize)
	{
		for (int i_idx = 0, j = 0; j < ff_osize; ++j, ++index, ++start)
		{
			i_idx = start;
			dW[-1] += dO[index]; // the bias

			for (int f_idx = 0, k = 0; k < ff_width; ++k, i_idx += stride)
				for (int l = 0; l < ff_width; ++l, ++f_idx, ++i_idx)
{
assert (f_idx < ff_W.N ());
assert (index < ff_isize * ff_isize);
assert (i_idx < ff_isize * ff_isize);
assert (isnan (dW[f_idx]) == 0);
assert (isnan (dO[index]) == 0);
assert (isnan (input[i_idx]) == 0);
					dW[f_idx] += dO[index] * input[i_idx];
}
			//      ∂L/∂f =   ∑   (∂L/∂O)  • ∂O/∂f
		}
	}
}

#endif // header inclusion

