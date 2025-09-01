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

#ifndef __RANT_TRANSFORMER_BLOCK__H__
#define __RANT_TRANSFORMER_BLOCK__H__

#include <AttentionHead.h>
#include <2D_normalization.h>
#include <ffn.h>

#define RELU(X) ((X) < 0.0 ? 0.0 : (X))
#define RELU_DERIVATIVE_FN(Y) (Y > 0.0 ? 1.0 : 0.0)

class transformerBlock_t
{
	int							tb_l;
	int							tb_d;

	Md_t						tb_Y;
	Md_t						tb_dX;

	Attention_t					tb_Attn;

#ifndef __NO_TRANSBLOCK_FFN
	ffn_t						tb_ffn1;
	ffn_t						tb_ffn2;
#endif

#ifndef __NO_LAYERNORM_
	MatrixNormalization_t		tb_Z0;
	MatrixNormalization_t		tb_Z1;
#endif

	Md_t						tb_dZ;

public:

	transformerBlock_t (const int h, const int l, const int d, bool causal) :
		tb_l (l),
		tb_d (d),
		tb_Y (tb_l, tb_d),
		tb_Attn (h, l, d, causal)
#ifndef __NO_TRANSBLOCK_FFN
		,tb_ffn1 (d, 2*d, true),
		tb_ffn2 (2*d, d)
#endif
#ifndef __NO_LAYERNORM_
		,tb_Z0 (l, d)
		,tb_Z1 (l, d)
#endif
	{
	}

	~transformerBlock_t (void)
	{
	}

	Md_t &call (Md_t &X)
	{
		tb_Y = tb_Attn.call (X);

#ifndef __NO_LAYERNORM_
		tb_Y = tb_Z0.call (tb_Y);
#endif // __NO_LAYERNORM_

#ifndef __NO_TRANSBLOCK_FFN
		Md_t H = tb_ffn1.call (tb_Y);
		tb_Y = tb_ffn2.call (H);
#endif // __NO_TRANSBLOCK_FFN

#ifndef __NO_LAYERNORM_
		tb_Y = tb_Z1.call (tb_Y);
#endif // __NO_LAYERNORM_

		return tb_Y;
	}

	Md_t &backward (Md_t &dL)
	{
#ifndef __NO_LAYERNORM_
		dL = tb_Z1.backward (dL);
#endif

#ifndef __NO_TRANSBLOCK_FFN
		Md_t dZ = tb_ffn2.backward (dL);
		dL = tb_ffn1.backward (dZ);
#endif

#ifndef __NO_LAYERNORM_
		dL = tb_Z0.backward (dL);
#endif
		tb_dX = tb_Attn.backward (dL);

		return tb_dX;
	}

	void update (void)
	{
		tb_Attn.update ();

#ifndef __NO_TRANSBLOCK_FFN
		tb_ffn1.update ();
		tb_ffn2.update ();
#endif

#ifndef __NO_LAYERNORM_
		tb_Z1.update ();
		tb_Z0.update ();
#endif
	}
};

#endif // header inclusion

