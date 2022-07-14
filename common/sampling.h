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

#ifndef __NEUROTIC_SAMPLING_SHUFFLE__H__
#define __NEUROTIC_SAMPLING_SHUFFLE__H__

#include <stdlib.h>

class NoReplacementSamples_t {

	int				is_N;
	int				is_cursor;
	int				*is_samples;

public:

	NoReplacementSamples_t (int N) :
		is_N (N),
		is_cursor (0),
		is_samples (new int [is_N])
	{
		for (int i = 0; i < is_N; ++i)
			is_samples[i] = i;

		Reset ();
	}

	~NoReplacementSamples_t (void)
	{
		delete [] is_samples;
	}

	void Reset (void)
	{
		int index;
		int tmp;
		int *base = is_samples;

		for (int i = 0; i < is_N; ++i, ++base)
		{
			index = rand () % (is_N - i);
			tmp = base[index];
			base[index] = *base;
			*base = tmp;
		}

		is_cursor = 0;
	}

	int Sample (void)
	{
		if (is_cursor == is_N)
			return -1;

		return is_samples[is_cursor++];
	}

	int SampleAuto (void)
	{
		if (is_cursor == is_N)
			Reset ();

		return Sample ();
	}

	int N (void) const
	{
		return is_N;
	}

	int * raw (void)
	{
		return is_samples;
	}
};

#endif // header inclusion

