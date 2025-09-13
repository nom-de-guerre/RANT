
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

#ifndef __RANT_LAYER__H__
#define __RANT_LAYER__H__

#include <list>

struct layer_t;

typedef std::list<layer_t *> callable_t;

struct layer_t
{
	callable_t			l_children;

	Md_t				l_Y;
	Md_t				l_dX;

	layer_t (void)
	{
	}

	virtual ~layer_t (void)
	{
		for (auto component = l_children.rbegin ();
				component != l_children.rend ();
				++component)
		delete *component;
	}

	virtual Md_t &call (Md_t &X)
	{
		Md_t A = X;

		for (auto component = l_children.begin ();
				component != l_children.end ();
				++component)
		{
			l_Y = (*component)->call (A);
			A = l_Y;
		}

		return l_Y;
	}

	virtual Md_t &backward (Md_t &dL)
	{
		Md_t G = dL;

		for (auto component = l_children.rbegin ();
				component != l_children.rend ();
				++component)
		{
			l_dX = (*component)->backward (G);
			G = l_dX;
		}

		return l_dX;
	}

	virtual void update (void)
	{
		for (auto component = l_children.begin ();
				component != l_children.end ();
				++component)
			(*component)->update ();
	}

	virtual int N_LearnableParameters (void) const
	{
		return 0;
	}
};

#endif // header inclusion

