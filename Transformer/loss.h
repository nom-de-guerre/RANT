
/*  
 
Copyright (c) 2025, Douglas Santry 
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

#ifndef __RANT_NPL_LOSS__H__
#define __RANT_NPL_LOSS__H__

#include <math.h>

#include <transformer.h>

#define __MASKED_SKIP_POSITION      -1

struct loss_t
{
	Md_t						lo_G;
	IEEE_t						lo_loss;
	IEEE_t						lo_accuracy;
	int							lo_N;

	loss_t (void)
	{
		reset ();
	}

	IEEE_t getLoss (void) const
	{
		return lo_loss / lo_N;
	}

	IEEE_t getAccuracy (void) const
	{
		return lo_accuracy / lo_N;
	}

	void reset (void)
	{
		lo_loss = lo_accuracy = 0;
		lo_N = 0;
	}

	virtual Md_t &loss (Md_t &S, int const * const y, int const * const _y) = 0;
};

class SparseCrossEntropy_t : public loss_t
{

public:

	SparseCrossEntropy_t (void) :
		loss_t ()
	{
	}

	virtual Md_t &loss (Md_t &S, int const * const y, int const * const _y)
	{
		int l = S.rows ();

		lo_G = S;
		lo_G.copy ();
		IEEE_t check;
		IEEE_t XE = 0.0;
		IEEE_t accuracy = 0.0;
		IEEE_t Ntokens = 0;

		for (int i = 0; i < l; ++i)
		{
			if (y[i] == __MASKED_SKIP_POSITION)
				continue;

			check = -log (lo_G (i, y[i]));
			if (!isnan (check))
				XE += check;

			if (y[i] == _y[i])
				++accuracy;

			lo_G (i, y[i]) -= 1;

			++Ntokens;
		}

		lo_loss += XE / Ntokens;
		assert (!isnan (lo_loss));
		lo_accuracy += accuracy / Ntokens;
		assert (!isnan (lo_accuracy));
		++lo_N;

		return lo_G;
	}
};

#endif // header inclusion

