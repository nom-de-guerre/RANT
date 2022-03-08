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

#include <CNN.h>

double Validate (CNN_t &model, DataSet_t *datap)
{
	int incorrect = 0;

	for (int i = 0; i < datap->N (); ++i)
	{
		plane_t obj (IMAGEDIM, IMAGEDIM, datap->entry (i));

		int k = model.Classify (&obj);

		if (k != datap->Answer (i))
			++incorrect;
	}

	double ratio = (double) incorrect;
	ratio /= (double) datap->t_N;
	ratio *= 100;

	return ratio;
}

void Validate (CNN_t &model, DataSet_t *datap, double &accuracy, double &loss)
{
	int incorrect = 0;
	int answer;

	SoftmaxNNm_t *softp = model.Bottom ();

	accuracy = 0;
	loss = 0;

	for (int i = 0; i < datap->N (); ++i)
	{
		plane_t obj (IMAGEDIM, IMAGEDIM, datap->entry (i));

		int k = model.Classify (&obj);

		answer = datap->Answer (i);
		if (k != answer)
			++incorrect;

		loss += -log (softp->P(answer));
	}

	double ratio = (double) incorrect;
	ratio /= (double) datap->t_N;
	ratio *= 100;

	accuracy = ratio;
	loss /= (double) datap->t_N;
}

