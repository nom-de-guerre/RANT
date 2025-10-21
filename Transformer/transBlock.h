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

class transformerBlock_t : public layer_t
{
public:

	transformerBlock_t (const int h, const int l, const int d, bool causal) :
		layer_t ()
	{
		l_children.push_back (new Attention_t (h, l, d, causal));
#ifndef __NO_LAYERNORM_
		l_children.push_back (new MatrixNormalization_t (l, d));
#endif
#ifndef __NO_TRANSBLOCK_FFN
		l_children.push_back (new ffn_t (d, 2*d, true));
		l_children.push_back (new ffn_t (2*d, d));
#endif
#ifndef __NO_LAYERNORM_
		l_children.push_back (new MatrixNormalization_t (l, d));
#endif
	}

	transformerBlock_t (FILE *fp)
	{
		char buffer[32];

		fscanf (fp, "%s\n", buffer);
		if (strcmp (buffer, "@TBLOCK") != 0)
			throw ("Bad Transformer Block");

		l_children.push_back (new Attention_t (fp));
		l_children.push_back (new MatrixNormalization_t (fp));
		l_children.push_back (new ffn_t (fp));
		l_children.push_back (new ffn_t (fp));
		l_children.push_back (new MatrixNormalization_t (fp));
	}

	~transformerBlock_t (void)
	{
	}

	virtual bool save (FILE *fp)
	{
		fprintf (fp, "@TBLOCK\n");
		layer_t::save (fp);

		return true;
	}
};

#endif // header inclusion

