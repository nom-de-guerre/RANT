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

#ifndef __RANT_MATRIX_LEARNABLE__H__
#define __RANT_MATRIX_LEARNABLE__H__

#include <stdlib.h>

#include <transformer_common.h>
#include <Optimizer.h>

/*
 *
 * Y = XW
 *
 */

class __matrix_XW_t
{
	Md_t				mp_W;
	Md_t				mp_dW; // error if learnable
	Md_t				mp_Y;

	Md_t				mp_X;
	Md_t				mp_dX; // gradient exiting the object

	bool				mp_training;             // weights are mutable

	Optimizer_t			mp_O;

	char const * const	mp_name;

public:

	__matrix_XW_t (int d, int n, const char * name = "anon") :
		mp_W (d, n),
		mp_dW (d, n),
		mp_dX (d, n),
		mp_training (true),
		mp_O (d, n, mp_W.raw (), mp_dW.raw ()),
		mp_name (strdup (name))
	{
		InitLearnable (mp_W.N (), d, mp_W.raw ());
		mp_dW.zero ();

	}

	~__matrix_XW_t (void)
	{
		free ((void *) mp_name);
	}

	char const *Id (void) const
	{
		return mp_name;
	}

	Md_t &call (Md_t &X)
	{
		if (mp_training)
			mp_X = X;

		mp_Y = X * mp_W;

		matrix_paranoia (mp_Y);

		return mp_Y;
	}

    /*
	 * M = XW
	 *
	 *
     *      ∂L   ∂L
     * dX = -- = -- · W^T
     *      ∂X   ∂M
	 *
     *      ∂L         ∂L
     * dW = -- = X^T · --
     *      ∂W         ∂M
	 *
	 *
	 */

	Md_t &backward (Md_t &G)
	{
		mp_dX = G * mp_W.transpose ();

		mp_dW += mp_X.transpose () * G;

		matrix_paranoia (mp_dX);
		matrix_paranoia (mp_dW);

		return mp_dX;
	}

	void update (void)
	{
		assert (mp_training);

		mp_O.update ();
	}

	Md_t &W (void)
	{
		return mp_W;
	}

	Md_t &dW (void)
	{
		return mp_dW;
	}

	Md_t &Y (void)
	{
		return mp_Y;
	}

	Md_t &X (void)
	{
		return mp_X;
	}

	void freeze (void)
	{
		mp_training = false;
	}

	int N_LearnableParameters (void) const
	{
		return mp_W.N ();
	}
};

/*
 * Tracks a matrix
 *
 */

struct __matrix_dX_t
{
	Md_t				mp_X;
	Md_t				mp_dX;				// gradient exiting the object

	bool				mp_training;		// weights are mutable

	Optimizer_t			mp_O;

	char const * const	mp_name;

	__matrix_dX_t (int n, int m, const char * name = "anon") :
		mp_X (n, m),
		mp_dX (n, m),
		mp_training (true),
		mp_O (n, m, mp_X.raw (), mp_dX.raw ()),
		mp_name (strdup (name))
	{
		mp_X.zero ();
		mp_dX.zero ();
	}

	~__matrix_dX_t (void)
	{
		free ((void *) mp_name);
	}

	char const *Id (void) const
	{
		return mp_name;
	}

	void update (void)
	{
		assert (mp_training);

		mp_O.update ();
	}
};

/*
 *
 * Z = AB
 *
 */
class __matrix_recorded_AB_t
{
	Md_t				mp_A;
	Md_t				mp_B;

	Md_t				mp_AB;

	Md_t				mp_dA;
	Md_t				mp_dB;

	bool				mp_training;

public:

	__matrix_recorded_AB_t () :
		mp_training (true)
	{
	}

	~__matrix_recorded_AB_t (void)
	{
	}

	Md_t &call (Md_t &A, Md_t &B)
	{
		if (mp_training)
		{
			mp_A = A;
			mp_B = B;
		}

		mp_AB = A*B;
		return mp_AB;
	}

    /*
	 * M = AB
	 *
	 *
     *      ∂L   ∂L
     * dA = -- = -- · B^T
     *      ∂A   ∂M
	 *
     *      ∂L         ∂L
     * dB = -- = A^T · --
     *      ∂B         ∂M
	 *
	 *
	 */

	void backward (Md_t &G)
	{
		mp_dA = G * mp_B.transpose ();

		mp_dB = mp_A.transpose () * G;
	}

	Md_t &dA (void)
	{
		return mp_dA;
	}

	Md_t &dB (void)
	{
		return mp_dB;
	}

	Md_t &Y (void)
	{
		return mp_AB;
	}
};

/*
 *
 * Z = AB^T
 *
 */
class __matrix_recorded_ABt_t
{
	Md_t				mp_A;
	Md_t				mp_B;
	Md_t				mp_ABt;

	Md_t				mp_dA;
	Md_t				mp_dB;

	bool				mp_training;

public:

	__matrix_recorded_ABt_t () :
		mp_training (true)
	{
	}

	~__matrix_recorded_ABt_t (void)
	{
	}

	Md_t &call (Md_t &A, Md_t &B)
	{
		if (mp_training)
		{
			mp_A = A;
			mp_B = B;
		}

		mp_ABt = A * B.transpose ();

		return mp_ABt;
	}

    /*
	 * M = AB^T
	 *
	 * Below accounts for transpose of B
	 *
     *      ∂L   ∂L
     * dA = -- = -- · B
     *      ∂X   ∂M
	 *
     *      ∂L         ∂L   ∂L^T
     * dB = -- = X^T · -- = ---- · A
     *      ∂W         ∂M   ∂M
	 *
	 *
	 */

	void backward (Md_t &G)
	{
		mp_dA = G * mp_B;

		mp_dB = G.transpose () * mp_A;
	}

	Md_t &dA (void)
	{
		return mp_dA;
	}

	Md_t &dB (void)
	{
		return mp_dB;
	}
};

#endif // header inclusion

