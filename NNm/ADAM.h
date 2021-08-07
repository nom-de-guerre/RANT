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

#ifndef _NN_ADAM_STRATEGY__H__
#define _NN_ADAM_STRATEGY__H__

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include <NeuralM.h>

#define ALPHA		0.001
#define BETA1		0.9
#define BETA2		0.999
#define EPSILON		1e-8

/*
 * ADAM strategy implementation.
 *
 */
struct ADAMStrategy_t : public stratum_t
{
	NeuralM_t				r_Mi;
	NeuralM_t               r_Vi;

	ADAMStrategy_t (const int N, const int Nin) :
		stratum_t (N, Nin),
		r_Mi (s_Nperceptrons, s_Nin),
		r_Vi (s_Nperceptrons, s_Nin)
	{
		r_Mi.zero ();
		r_Vi.zero ();
	}

	~ADAMStrategy_t (void)
	{
	}

	void Strategy (void);
	void ADAM (int);
};

void 
ADAMStrategy_t::Strategy (void)
{
	int Nweights = s_Nperceptrons * s_Nin;

	for (int index = 0; index < Nweights; ++index)
		ADAM (index);
}

void 
ADAMStrategy_t::ADAM (int index)
{
	double g = s_dL.sm_data[index];
	double m;
	double v;
	double update;

	r_Mi.sm_data[index] = BETA1 * r_Mi.sm_data[index] + (1 - BETA1) * g;
	r_Vi.sm_data[index] = BETA2 * r_Vi.sm_data[index] + (1 - BETA2) * (g * g);
	m = r_Mi.sm_data[index] / (1 - BETA1);
	v = r_Vi.sm_data[index] / (1 - BETA2);
	update = m / (sqrt (v) + EPSILON);

	s_W.sm_data[index] -= ALPHA * update;

	s_dL.sm_data[index] = 0.0;
}

stratum_t * AllocateADAM (const int N, const int Nin);

stratum_t * AllocateADAM (const int N, const int Nin)
{
	stratum_t *p = new ADAMStrategy_t (N, Nin);

	return p;
}

#define ADAM AllocateADAM

#endif // header inclusion

