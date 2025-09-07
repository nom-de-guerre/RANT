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

#ifndef __RANT_SOFTMAX_2D__H__
#define __RANT_SOFTMAX_2D__H__

#include <math.h>

#include <transformer_common.h>

class Softmax2D_t : public layer_t
{
	Md_t		ss_S;
	Md_t		ss_G;
	int			*ss_indices;

	bool		ss_applyCausalMask;

public:

	Softmax2D_t (const int rows, const int columns, const bool causal=false) :
		ss_indices (new int [rows]),
		ss_applyCausalMask (causal)
	{
	}

	~Softmax2D_t (void)
	{
		delete [] ss_indices;
	}

	inline Md_t &S (void)
	{
		return ss_S;
	}

	virtual Md_t &call (Md_t &X)
	{
		int rows = X.rows ();
		int columns = X.columns ();
		ss_S = Md_t (rows, columns);

        IEEE_t Sdenom;
		IEEE_t translate;

		/*
		 * Per row softmax
		 *
		 */

		const int Xstride = X.stride ();
		const int Sstride = ss_S.stride ();

        for (int i = 0; i < rows; ++i)
        {
        	IEEE_t const * __restrict Xp = X.raw () + i;
        	IEEE_t * __restrict Sp = ss_S.raw () + i;

            Sdenom = 0.0;
			translate = Xp[0];

			for (int j = 0; j < columns; ++j, Sp += Sstride, Xp += Xstride)
			{
				if (ss_applyCausalMask && j == (i + 1))
					break;

				*Sp = *Xp;
				if (*Sp > translate)
					translate = *Sp;
			}

			Sp = ss_S.raw () + i;
			for (int j = 0; j < columns; ++j, Sp += Sstride)
			{
				if (ss_applyCausalMask && j == (i + 1))
					break;

				*Sp = exp (*Sp - translate);
				Sdenom += *Sp;
				assert (isnan (Sdenom) == 0);
			}

			Sp = ss_S.raw () + i;
            for (int j = 0; j < columns; ++j, Sp += Sstride)
			{			
				if (ss_applyCausalMask && j > i)
					*Sp = 0.0;
				else
					*Sp /= Sdenom;
			}
        }

#ifdef __DEBUG_SOFTMAX
		matrix_paranoia (ss_S);

		for (int i = 0; i < rows; ++i)
		{
			IEEE_t __verify = 0.0;

			for (int j = 0; j < columns; ++j)
			{
				IEEE_t element = ss_S (i, j);
				assert (element >= 0 && element <= 1.0);

				__verify += element;
			}

			assert (__verify <= 1.02 && __verify > 0.98);
		}
#endif

		return ss_S;
	}

	/*
	 * There are no learnable parameters.  The sole function is
	 * to compute the exiting gradient.
	 *
	 */
	virtual Md_t &backward (Md_t &dL)
	{
		const int rows = ss_S.rows ();
		const int columns = ss_S.columns ();
		ss_G = Md_t (rows, columns);

		for (int i = 0; i < rows; ++i)
		{
			Md_t dlds = dL.view (i, 0, 1, columns);
			Md_t s = ss_S.view (i, 0, 1, columns);

			Md_t J = ComputeJacobian (s);

			Md_t du = dlds * J;

			ss_G.importRow (i, du.raw ());
		}

		return ss_G;
	}

	// Construct Ji.  Note that it is symetric.
	Md_t ComputeJacobian (Md_t &s)
	{
		int rows = ss_S.rows ();
		const int stride = ss_S.stride ();
		Md_t J = s.transpose () * s;
        J *= -1;

		IEEE_t * __restrict Jp = J.raw ();

		for (int i = 0; i < rows; ++i, Jp += (stride + 1))
			*Jp += s(0, i);

		return J;
	}

	int *argmax (void)
	{
		int rows = ss_S.rows ();
		int columns = ss_S.columns ();

		for (int i = 0; i < rows; ++i)
		{
			IEEE_t min = -DBL_MIN;
			int index = 0;
			for (int j = 0; j < columns; ++j)
			{
				if (ss_S (i, j) > min)
				{
					min = ss_S (i, j);
					index = j;
				}
			}

			ss_indices[i] = index;
		}

		return ss_indices;
	}
};

#endif // header inclusion

