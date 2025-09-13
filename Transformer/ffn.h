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

#ifndef __RANT_FFN_COMPONENT__H__
#define __RANT_FFN_COMPONENT__H__

#include <transformer_common.h>

#define RELU(X) ((X) < 0.0 ? 0.0 : (X))
#define RELU_DERIVATIVE_FN(Y) (Y > 0.0 ? 1.0 : 0.0)

class ffn_t : public layer_t
{
	__matrix_XW_t				f_ffn;

	Md_t						f_Y;
	Md_t						f_dX;

	bool						f_activate;

public:

	ffn_t (const int rows, const int columns, bool activate=false) :
		layer_t (),
		f_ffn (rows, columns),
		f_activate (activate)
	{
	}

	~ffn_t (void)
	{
	}

	virtual Md_t &call (Md_t &X)
	{
		f_Y = f_ffn.call (X);

		if (f_activate)
		{
			f_Y.copy ();

			int N = f_Y.N ();
			IEEE_t * __restrict p = f_Y.raw ();
			for (int i=0; i < N; ++i, ++p)
				*p = RELU (*p);
		}

		return f_Y;
	}

	virtual Md_t &backward (Md_t &dL)
	{
		if (f_activate)
		{
			IEEE_t * __restrict p = f_Y.raw ();
			IEEE_t * __restrict q = dL.raw ();
			int N = dL.N ();

			for (int index = 0; index < N; ++index, ++p, ++q)
				*q *= RELU_DERIVATIVE_FN (*p);
		}

		f_dX = f_ffn.backward (dL);

		return f_dX;
	}

	virtual void update (void)
	{
		f_ffn.update ();
	}

	virtual int N_LearnableParameters (void) const
	{
		return f_ffn.N_LearnableParameters ();
	}
};

#endif // header inclusion

