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
	NeuralM_t				ad_Mi;
	NeuralM_t               ad_Vi;
	double					ad_beta1;
	double 					ad_beta2;

	ADAMStrategy_t (const int N, const int Nin) :
		stratum_t (N, Nin),
		ad_Mi (s_Nperceptrons, s_Nin),
		ad_Vi (s_Nperceptrons, s_Nin),
		ad_beta1 (BETA1),
		ad_beta2 (BETA2)
	{
		ad_Mi.zero ();
		ad_Vi.zero ();
	}

	~ADAMStrategy_t (void)
	{
	}

	void Strategy (void);
	inline void ADAM (int);
};

void 
ADAMStrategy_t::Strategy (void)
{
	int Nweights = ad_Mi.N ();

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

	ad_Mi.sm_data[index] = BETA1 * ad_Mi.sm_data[index] + (1 - BETA1) * g;
	ad_Vi.sm_data[index] = BETA2 * ad_Vi.sm_data[index] + (1 - BETA2) * (g * g);
	m = ad_Mi.sm_data[index] / (1 - ad_beta1);
	v = ad_Vi.sm_data[index] / (1 - ad_beta2);
	update = m / (sqrt (v) + EPSILON);

	s_W.sm_data[index] -= ALPHA * update;
	ad_beta1 *= BETA1;
	ad_beta2 *= BETA2;

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

