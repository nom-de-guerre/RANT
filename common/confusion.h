/*

Copyright (c) 2022, Douglas Santry
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

#ifndef __NNm_CONFUSION_HEADER__H_
#define __NNm_CONFUSION_HEADER__H_

#include <data.h>
#include <NeuralM.h>

class confusion_t : public NeuralM_t
{
	int						cm_K;
	int						cm_N;
	int						**cm_stats;

public:

	enum sType_e { Ki,
					TruePositives,
					FalsePositives,
					TrueNegatives,
					FalseNegatives };

	confusion_t (DataSet_t *O, NNet_t *Mp) :
		NeuralM_t (Mp->Nout (), Mp->Nout ()),
		cm_K (Mp->Nout ()),
		cm_N (O->N ())
	{
		cm_stats = new int * [cm_K];

		for (int i = 0; i < cm_K; ++i)
			cm_stats[i] = new int [5];

		for (int i = 0; i < cm_K; ++i)
			cm_stats[i][Ki] = 0;

		zero ();

		Fill (O, Mp);
		ComputeMeasures ();
	}

	confusion_t (const int K) :
		NeuralM_t (K, K),
		cm_K (K),
		cm_N (0)
	{
		cm_stats = new int * [cm_K];

		for (int i = 0; i < cm_K; ++i)
			cm_stats[i] = new int [5];

		for (int i = 0; i < cm_K; ++i)
			cm_stats[i][Ki] = 0;

		zero ();
	}

	~confusion_t (void)
	{
		for (int i = 0; i < cm_K; ++i)
			delete [] cm_stats[i];

		delete [] cm_stats;
	}

	void Update (DataSet_t *O, NNet_t *Mp)
	{
		cm_N += O->N ();

		Fill (O, Mp);
		ComputeMeasures ();
	}

	int GetMeasure (int k, sType_e measure) const
	{
		if (k < 0 || k >= cm_K)
			return -1;

		return cm_stats[k][measure];
	}

	int GetTP (const int k) const
	{
		return GetMeasure (k, confusion_t::TruePositives);
	}

	int GetFP (const int k) const
	{
		return GetMeasure (k, confusion_t::FalsePositives);
	}

	int GetTN (const int k) const
	{
		return GetMeasure (k, confusion_t::TrueNegatives);
	}

	int GetFN (const int k) const
	{
		return GetMeasure (k, confusion_t::FalseNegatives);
	}

	int NumberCorrect (void) const
	{
		int correct = 0;

		for (int i = 0; i < cm_K; ++i)
			correct += GetTP (i);

		return correct;
	}

	float ratioCorrect (void) const
	{
		return (float) NumberCorrect () / (float) cm_N;
	}

	void DumpStats (void)  const
	{
		printf ("\tTP\tFP\tTN\tFN\n");

		for (int i = 0; i < cm_K; ++i)
			printf ("(%d)\t%d\t%d\t%d\t%d\n",
				i,
				GetTP (i),
				GetFP (i),
				GetTN (i),
				GetFN (i));
	}

private:

	void Fill (DataSet_t *O, NNet_t *Mp)
	{
		for (int i = 0; i < O->N (); ++i)
		{
			int k = (int) O->Answer (i);
			int guess = (int) Mp->Compute ((*O)[i]);

			cm_stats[k][Ki]++;

			Mentry (guess, k) += 1;
		}
	}

	void ComputeMeasures (void)
	{
		for (int i = 0; i < cm_K; ++i)
			ComputeMeasures_k (i);
	}

	void ComputeMeasures_k (const int k)
	{
		for (int i = Ki; i < 4; ++i)
			cm_stats[k][i] = 0;

		cm_stats[k][TruePositives] = Mentry (k, k);

		for (int i = 0; i < cm_K; ++i)
		{
			if (i == k)
				continue;

			cm_stats[k][FalsePositives] += Mentry (k, i);
			cm_stats[k][FalseNegatives] += Mentry (i, k);
		}

		cm_stats[k][TrueNegatives] = cm_N -
			cm_stats[k][TruePositives] -
			cm_stats[k][FalsePositives] -
			cm_stats[k][FalseNegatives];
	}
};

#endif // header inclusion

