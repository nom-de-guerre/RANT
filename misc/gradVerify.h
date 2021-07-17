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

#ifndef __NEUROTIC_GRADVERIFY__H__
#define __NEUROTIC_GRADVERIFY__H__

#include <CNN.h>

class Gradient_t : public CNN_t
{

	plane_t					*cg_example;
	double					cg_answer;
	full_t					*cg_softmaxp;

public:

	Gradient_t (const int rows,
		const int columns,
		const int Nlayers,
		const int Nclasses) : CNN_t (rows, columns, Nlayers, Nclasses)
	{
	}

	void f_diff (int level)
	{
		if (!level)
		{
			cn_layers[0]->f (cg_example);
			++level;
		}
		
		for (int i = level; i < cn_N; ++i)
			cn_layers[i]->f (cn_layers[i - 1]);
	}

	void VerifyGradientWork (int level, double h)
	{
		plane_t *Xi = (level ? (*cn_layers[level - 1])[0] : cg_example);
		int Nentries = Xi->N ();
		double *rawp = Xi->raw ();
		plane_t *gradp;
		double L0, L1;
		double dL_diff, dL_bp;

		for (int i = 0; i < Nentries; ++i)
		{
			rawp[i] += h;

			cg_softmaxp->Cycle ();
//			f_diff (level);
Classify (cg_example);
			L0 = -log (cg_softmaxp->P ((int) cg_answer));

			rawp[i] -= 2 * h;

			cg_softmaxp->Cycle ();
	//		f_diff (level);
Classify (cg_example);
			L1 = -log (cg_softmaxp->P ((int) cg_answer));

			dL_diff = (L0 - L1) / (2 * h);

			rawp[i] += h;

			cg_softmaxp->Cycle ();
			TrainExample (*cg_example, cg_answer);

			gradp = cn_layers[level]->getModule (0)->fetchGradient ();

			dL_bp = gradp->raw () [i];

			printf ("DJS\t%f\t%f\t%f\n",
				dL_diff,
				dL_bp,
				fabs (dL_diff - dL_bp) / (fabs (dL_diff) + fabs (dL_bp)));
		}
	}

	void VerifyGradient (int level, double h, plane_t Xi, double answer)
	{
		assert (level < cn_N);
		cg_example = &Xi;
		cg_answer = answer;
		cg_softmaxp = cn_layers[cn_N - 1]->Bottom ();

		if (level)
			cn_layers[0]->f (&Xi);
		
		for (int i = 1; i < level; ++i)
			cn_layers[i]->f (cn_layers[i - 1]);

		VerifyGradientWork (level, h);
	}
};

#endif // header inclusion

