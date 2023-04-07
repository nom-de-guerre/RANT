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

#ifndef _NNm_IDENTITY__H__
#define _NNm_IDENTITY__H__

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include <NeuralM.h>
#include <NNm.h>

/*
 * The identity layer - just passes signals through regardless of direction.
 *
 */
struct identity_t : public stratum_t
{
	identity_t (const int ID, const int N) : stratum_t ("I", ID, N, N)
	{
	}

	virtual ~identity_t (void)
	{
	}

	void _sAPI_init (void)
	{
	}

	virtual IEEE_t * _sAPI_f (IEEE_t * const, bool = true);
	virtual void _sAPI_gradient (stratum_t &);
	virtual void _sAPI_bprop (IEEE_t *, bool = true);

	virtual void StrategyMono (const int index)
	{
		return; // over-ride when debugging or instrumenting
	}
};

void
identity_t::_sAPI_gradient (stratum_t &Z)
{
	Z.s_delta.Accept (s_delta.raw ());
}

void 
identity_t::_sAPI_bprop (IEEE_t *xi, bool activation)
{
}

IEEE_t *
identity_t::_sAPI_f (IEEE_t * const xi, bool activate)
{
	s_response.Accept (xi);

	return s_response.sm_data;
}

#endif // header inclusion

