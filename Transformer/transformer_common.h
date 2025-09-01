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

#ifndef __RANT_TRANSFORMER_COMMON__H__
#define __RANT_TRANSFORMER_COMMON__H__

#include <assert.h>

typedef double IEEE_t;
#include <matrix.h>
typedef Matrix_t<double> Md_t;

void InitLearnable (const int N, const int fanIn, IEEE_t *learnable)
{
	IEEE_t r = sqrt (6.0 / (2 * fanIn));
	IEEE_t sample;
	IEEE_t *p = learnable;

	for (int i = 0; i < N; ++i)
	{
		sample = (IEEE_t) rand () / RAND_MAX;
		sample *= r;

		if (rand () % 2)
			sample = -sample;
		*p++ = sample;
	}
}

#include <layer.h>

struct counter_t
{
	IEEE_t			c_agg;
	IEEE_t			c_N;

	counter_t (void)
	{
		reset ();
	}

	void operator+= (IEEE_t x)
	{
		c_agg += x;
		++c_N;
	}

	IEEE_t get (void) const
	{
		return c_agg / c_N;
	}

	void reset (void)
	{
		c_agg = c_N = 0.0;
	}
};

#endif // header inclusion

