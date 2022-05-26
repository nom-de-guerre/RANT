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

#ifndef _NN_STRATUM__H__
#define _NN_STRATUM__H__

#include <NeuralM.h>
#include <strategy.h>

struct stratum_t
{
	const int				s_ID;

	int						s_Nnodes;
	int						s_Nin;

	NeuralM_t				s_delta;
	NeuralM_t				s_response;

	strategy_t				*s_strat;

	stratum_t (const int ID, const int N, const int Nin) :
		s_ID (ID),
		s_Nnodes (N),
		s_Nin (Nin),
		s_delta (N, 1),
		s_response (N, 1)
	{
	}

	virtual ~stratum_t (void)
	{
	}

	int N (void) const
	{
		return s_Nnodes;
	}

	virtual void _sAPI_init (const int) = 0;
	virtual IEEE_t * _sAPI_f (IEEE_t * const, bool = true) = 0;

	virtual void _sAPI_gradient (stratum_t &) = 0;
	virtual void _sAPI_bprop (IEEE_t *, bool = true) = 0;

	virtual void _sAPI_strategy (void)
	{
		s_strat->_tAPI_strategy ();
	}

	virtual NeuralM_t * _sAPI_gradientM (void)
	{
		return &s_delta;
	}

	IEEE_t * z (void)
	{
		return s_response.raw ();
	}
};

#endif // header inclusion

