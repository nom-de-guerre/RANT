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
			level = 1;
		}

		for (int i = level; i < cn_N; ++i)
			cn_layers[i]->f (cn_layers[i - 1]);
	}

	void VerifyGradient (int level, double h, plane_t &Xi, double answer)
	{
		assert (level >= 0 && level < cn_N);

		cg_example = &Xi;
		cg_answer = answer;
		cg_softmaxp = cn_layers[cn_N - 1]->Bottom ();

		plane_t BPROP = *ComputeBPROP (level);
		int Nentries = BPROP.N ();
		double *dL_bprop_flux = BPROP.raw ();

		printf ("Level %d\tDim\t%d\n", level, Nentries);

		double dL_diff;
		double dL_bp;
		double ratio;
		double denom;

		printf ("\tBPROP\t\tDiff\t\tRatio\n");

		for (int i = 0; i < Nentries; ++i)
		{
			dL_diff = ComputeDifference (level, i, h);
			dL_bp = dL_bprop_flux[i];

			denom = fabs (dL_diff) + fabs (dL_bp);
			ratio = (fabs (dL_diff) - fabs (dL_bp)) / denom;

			printf ("%d\t%f\t%f\t%f\t%s\n",
				i,
				dL_bp,
				dL_diff,
				ratio,
				(fabs (ratio) < 1e-4 ? "" : "X"));
		}
	}

	/*
	 * This will compare the gradient computed with BPROP at the
	 * specified level with differencing.
	 *
	 * When using differencing beware of disconinuities (e.g. max pooling).
	 *
	 */
	double ComputeDifference (int level, int entry, double h)
	{
		plane_t *Yi = (level ? 
			(*cn_layers[level - 1])[0] :
			cg_example);
		double *rawp = Yi->raw ();
		double save_entry;
		double dL_diff;
		double L0, L1;

		if (level)
			cn_layers[0]->f (cg_example);
		
		for (int i = 1; i <= level; ++i)
			cn_layers[i]->f (cn_layers[i - 1]);

		save_entry = rawp[entry];

		rawp[entry] += h;

		cg_softmaxp->Cycle ();
		f_diff (level);
		L0 = -log (cg_softmaxp->P ((int) cg_answer));

		rawp[entry] -= 2 * h;

		cg_softmaxp->Cycle ();
		f_diff (level);
		L1 = -log (cg_softmaxp->P ((int) cg_answer));

		dL_diff = (L0 - L1) / (2 * h);

		rawp[entry] = save_entry; // safer, no odd effects.

		return dL_diff;
	}

	plane_t *ComputeBPROP (int level)
	{
		cg_softmaxp->Cycle ();
		plane_t *Gp = cn_layers[level]->getModule (0)->fetchGradient ();
		Gp->Reset ();

		TrainExample (*cg_example, cg_answer);

		return cn_layers[level]->getModule (0)->fetchGradient ();
	}
};

#endif // header inclusion

