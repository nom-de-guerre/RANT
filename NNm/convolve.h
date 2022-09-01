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

#ifndef _NN_CONVOLUTION__H__
#define _NN_CONVOLUTION__H__

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include <NNm.h>
#include <filter.h>

/*
 * Implements a layer when training a neural network.
 *
 */

struct convolve_t : public stratum_t
{
	filter_t				**cn_filters;

	int						cn_filter;

	int						cn_imapSize;
	bool					cn_oneToOne;

	convolve_t (const int ID,
				const int N,
				const int fwidth,
				const shape_t Xin,
				StrategyAlloc_t rule) :
		stratum_t ("convolv",
				ID,
				shape_t (N, 
						Xin.sh_rows - fwidth + 1,
						Xin.sh_columns - fwidth + 1)),
		cn_filter (fwidth),
		cn_imapSize (Xin.mapSize ()),
		cn_oneToOne (1 == Xin.sh_N)
	{
		cn_filters = new filter_t * [sh_N];

		for (int i = 0; i < sh_N; ++i)
			cn_filters[i] = new filter_t (fwidth, Xin.sh_rows, rule);
	}

	virtual ~convolve_t (void)
	{
		for (int i = 0; i < sh_N; ++i)
			delete cn_filters[i];

		delete cn_filters;
	}

	int N (void) const
	{
		return sh_length;
	}

	void _sAPI_init (void)
	{
		// done in the filter_t constructor
	}

	virtual IEEE_t * _sAPI_f (IEEE_t * const, bool = true);
	virtual void _sAPI_gradient (stratum_t &);
	virtual void _sAPI_bprop (IEEE_t *, bool = true);

	virtual void _sAPI_strategy (void)
	{
		for (int i = 0; i < sh_N; ++i)
			cn_filters[i]->UpdateWeights ();
	}

	virtual int _sAPI_Trainable (void)
	{
		return sh_N * cn_filter * cn_filter + sh_N;
	}

	virtual void StrategyMono (const int index)
	{
		return; // over-ride when debugging or instrumenting
	}

	void DumpMaps (void)
	{
		s_response.display ();
	}
};

void
convolve_t::_sAPI_gradient (stratum_t &Z)
{
	IEEE_t *targetp = Z.s_delta.raw ();
	IEEE_t *gradp = s_delta.raw ();

	for (int i = 0; i < sh_N; ++i)
	{
		cn_filters[i]->Propagate (gradp, targetp);

		gradp += mapSize ();
		targetp += cn_imapSize;
	}
}

void 
convolve_t::_sAPI_bprop (IEEE_t *xi, bool activation)
{
	/*
	 *
	 * We are called with,
	 *
	 *    ∂L
	 *  ∑ -- ,
	 *    ∂z
	 *
	 * in s_delta.
	 *
	 */

	IEEE_t *inputp = xi;
	IEEE_t *gradp = s_delta.raw ();
	int block = mapSize ();

	for (int i = 0; i < sh_N; ++i)
	{
		cn_filters[i]->bprop (gradp, inputp);

		gradp += block;
		inputp += cn_imapSize;
	}
}

IEEE_t *
convolve_t::_sAPI_f (IEEE_t * const xi, bool activate)
{
	IEEE_t *inputp = xi;
	IEEE_t *outputp = s_response.raw ();
	int block = sh_rows * sh_columns;

	for (int i = 0; i < sh_N; ++i)
	{
		cn_filters[i]->f (inputp, outputp);

		outputp += block;
		if (!cn_oneToOne)
			inputp += cn_imapSize;
	}

	return s_response.sm_data;
}

#endif // header inclusion

