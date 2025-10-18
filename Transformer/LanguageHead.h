
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

#ifndef __RANT_LANGUAGE_HEAD__H__
#define __RANT_LANGUAGE_HEAD__H__

#include <transformer_common.h>
#include <LearnableMatrix.h>
#include <2D_softmax.h>
#include <layer.h>

class LanguageHead_t : layer_t
{

	__matrix_XW_t				lh_head;
	Softmax2D_t					lh_Y;

	Md_t						lh_logits;
	Md_t						lh_dZ;

public:

	LanguageHead_t (const int d, const int V) :
		layer_t (),
		lh_head (d, V),
		lh_Y (d, V)
	{
	}

	~LanguageHead_t (void)
	{
	}

	virtual Md_t &call (Md_t &Z)
	{
		/*
		 * The default representation is column-order.  Re-ordering
		 * the left-side of the equation results in cache line loads
		 * for the dot products.
		 *
		 */
		Md_t ZrowOrder (Z.rows (), Z.columns ());
		ZrowOrder.toRowOrder (Z);

		lh_logits = lh_head.call (ZrowOrder);
		return lh_Y.call (lh_logits);
	}

	/*
	 * Assumes that dS has been computed so lh_Y
	 * is skipped. The softmax step is multiclass, so
	 * J is not required.
	 *
	 */
	virtual Md_t &backward (Md_t &dS)
	{
		return lh_head.backward (dS);
	}

	virtual void update (void)
	{
		lh_head.update ();
	}

	int const * tokens (void)
	{
		return lh_Y.argmax ();
	}

	Md_t &S (void)
	{
		return lh_Y.S ();
	}

	Md_t &logits (void)
	{
		return lh_logits;
	}

	int N_LearnableParameters (void) const
	{
		return lh_head.N_LearnableParameters ();
	}
};

#endif // header inclusion

