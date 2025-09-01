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

#ifndef __RANT_ADAM_OPTIMIZER__H__
#define __RANT_ADAM_OPTIMIZER__H__

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include <transformer_common.h>

#define ALPHA		0.001
#define BETA1		0.9
#define BETA2		0.999
#define EPSILON		1e-7

/*
 * ADAM strategy implementation.
 *
 */

struct Optimizer_t
{
	Md_t				adam_Mi;
	Md_t                adam_Vi;

	IEEE_t				adam_beta1;
	IEEE_t 				adam_beta2;

	IEEE_t				*adam_learnable;
	IEEE_t				*adam_dL;

public:

	Optimizer_t (const int N, const int Nin, IEEE_t *W, IEEE_t *dL) :
		adam_Mi (N, Nin),
		adam_Vi (N, Nin),
		adam_beta1 (BETA1),
		adam_beta2 (BETA2),
		adam_learnable (W),
		adam_dL (dL)
	{
		adam_Mi.zero ();
		adam_Vi.zero ();
	}

	Optimizer_t (void) :
		adam_beta1 (BETA1),
		adam_beta2 (BETA2),
		adam_learnable (NULL),
		adam_dL (NULL)
	{
	}

	~Optimizer_t (void)
	{
	}

	void update (void);

	inline void ADAM (int);
};

void 
Optimizer_t::update (void)
{
	int Nweights = adam_Mi.N ();

	for (int index = 0; index < Nweights; ++index)
		ADAM (index);

	adam_beta1 *= BETA1;
	adam_beta2 *= BETA2;
}

void 
Optimizer_t::ADAM (int index)
{
	IEEE_t g = adam_dL[index];
	IEEE_t m;
	IEEE_t v;
	IEEE_t update;
	IEEE_t *Mi = adam_Mi.raw () + index;
	IEEE_t *Vi = adam_Vi.raw () + index;

	*Mi = BETA1 * *Mi + (1 - BETA1) * g;
	*Vi = BETA2 * *Vi + (1 - BETA2) * (g * g);
	m = *Mi / (1 - adam_beta1);
	v = *Vi / (1 - adam_beta2);
	update = m / (sqrt (v) + EPSILON);

	adam_learnable[index] -= ALPHA * update;
	assert (!isnan (adam_learnable[index]));

	adam_dL[index] = 0.0;
}

#endif // header inclusion

