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

#ifndef __RANT_TRANSFORMER__H__
#define __RANT_TRANSFORMER__H__

#include <transBlock.h>

class transformer_t
{
	int							t_Nblocks;
	transformerBlock_t			**t_blocks;

	Md_t						t_Z;
	Md_t						t_dX;

public:

	transformer_t (int N, int h, int l, int d, bool causal=false):
		t_Nblocks (N),
		t_blocks (new transformerBlock_t * [N])
	{
		for (int i = 0; i < t_Nblocks; ++i)
			t_blocks[i] = new transformerBlock_t (h, l, d, causal);
	}

	~transformer_t (void)
	{
		for (int i = 0; i < t_Nblocks; ++i)
			delete t_blocks[i];

		delete [] t_blocks;
	}

	Md_t &call (Md_t &X)
	{
		t_Z = X;

		for (int i=0; i < t_Nblocks; ++i)
			t_Z = t_blocks[i]->call (t_Z);

		return t_Z;
	}

	Md_t &backward (Md_t &G)
	{
		t_dX = G;

		for (int i = t_Nblocks - 1; i > -1; --i)
			t_dX = t_blocks[i]->backward (t_dX);

		return t_dX;
	}

	void update (void)
	{
		for (int i = t_Nblocks - 1; i > -1; --i)
			t_blocks[i]->update ();
	}
};

#endif // header inclusion

