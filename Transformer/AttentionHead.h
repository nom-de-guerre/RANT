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

#ifndef __RANT_ATTENTION_HEAD__H__
#define __RANT_ATTENTION_HEAD__H__

#include <transformer_common.h>
#include <LearnableMatrix.h>
#include <2D_softmax.h>

struct AttentionHead_t
{
	int								at_Id;

	__matrix_XW_t					at_Q;
	__matrix_XW_t					at_K;
	__matrix_XW_t					at_V;

	__matrix_recorded_ABt_t			at_QKt;

	Softmax2D_t						at_Softmax;

	__matrix_recorded_AB_t			at_head;

	Md_t							at_X;
	Md_t							at_dX;

	bool							at_causal;

public:

	/*
	 * d - embedding dimension
	 * dh - hidden dimension
	 *
	 */
	AttentionHead_t (int headId, 
					 const int l, 
					 const int d, 
					 const int dh, 
					 const bool causal) :
		at_Id (headId),
		at_Q (d, dh),
		at_K (d, dh),
		at_V (d, dh),
		at_Softmax (l, l),
		at_causal (causal)
	{
	}

	virtual ~AttentionHead_t (void)
	{
	}

	virtual Md_t &call (Md_t &X)
	{
		at_X = X;

		Md_t Q = at_Q.call (X);
		Md_t K = at_K.call (X);
		Md_t V = at_V.call (X);

		Md_t QKt = at_QKt.call (Q, K);

		Md_t S = at_Softmax.call (QKt);
		// Apply causal mask

		return at_head.call (S, V);
	}

	virtual Md_t &backward (Md_t &G)
	{
		at_head.backward (G);
		at_dX = at_V.backward (at_head.dB ());

		Md_t dS = at_Softmax.backward (at_head.dA ());
		at_QKt.backward (dS);

		at_dX += at_Q.backward (at_QKt.dA ());
		at_dX += at_K.backward (at_QKt.dB ());

		return at_dX;
	}

	virtual void update (void)
	{
		at_V.update ();
		at_K.update ();
		at_Q.update ();
	}

	Md_t &Y (void)
	{
		return at_head.Y ();
	}
};

class Attention_t : layer_t
{
	int							a_h;

	AttentionHead_t             **a_heads;
	__matrix_XW_t               **a_Wo;

	Md_t						a_Y;
	Md_t						a_dX;

public:

	Attention_t (const int h, 
				 const int l, 
				 const int d, 
				 const bool causal=false) :
		layer_t (),
		a_h (h),
		a_heads (new AttentionHead_t * [h]),
		a_Wo (new __matrix_XW_t * [h])
	{
		int dh = d / h;
		assert ((d % h) == 0);

		for (int i = 0; i < a_h; ++i)
		{
			a_heads[i] = new AttentionHead_t (i, l, d, dh, causal);
			a_Wo[i] = new __matrix_XW_t (dh, d, "Wo");
		}
	}

	virtual ~Attention_t (void)
	{
		for (int i = 0; i < a_h; ++i)
		{
			delete a_Wo[i];
			delete a_heads[i];
		}

		delete [] a_Wo;
		delete [] a_heads;
	}

	virtual Md_t &call (Md_t &X)
	{
		a_Y = Md_t (X.rows (), X.columns (), 0.0);

		for (int i = 0; i < a_h; ++i)
		{
			Md_t &Zi = a_heads[i]->call (X);
			a_Y += a_Wo[i]->call (Zi);
		}

		return a_Y;
	}

	virtual Md_t &backward (Md_t &dL)
	{
		a_dX = Md_t (dL.rows (), dL.columns (), 0.0);

		for (int i = 0; i < a_h; ++i)
		{
			Md_t G = a_Wo[i]->backward (dL);
			a_dX += a_heads[i]->backward (G);
		}

		return a_dX;
	}

	virtual void update (void)
	{
		for (int i = 0; i < a_h; ++i)
		{
			a_Wo[i]->update ();
			a_heads[i]->update ();
		}
	}
};

#endif // header inclusion

