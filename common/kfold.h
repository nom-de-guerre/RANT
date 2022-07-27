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

#ifndef __NNm_KFOLD_HEADER__H_
#define __NNm_KFOLD_HEADER__H_

#include <data.h>
#include <confusion.h>
#include <sampling.h>

class kfold_t
{
	DataSet_t * const				kf_data;

	DataSet_t						*kf_train;
	DataSet_t						*kf_test;

	NoReplacementSamples_t			kf_shuffle;

	float							kf_ratio;

public:

	kfold_t (DataSet_t *datap, float ratio = 0.667) :
		kf_data (datap),
		kf_train (NULL),
		kf_test (NULL),
		kf_shuffle (datap->N ()),
		kf_ratio (ratio)
	{
	}

	~kfold_t (void)
	{
		if (kf_train)
			delete kf_train;

		if (kf_test)
			delete kf_test;
	}

	void foldData (void)
	{
		int Ntrain = kf_data->N () * kf_ratio;
		int Ntest = kf_data->N () - Ntrain;

		kf_shuffle.Reset ();

		int const * const shuffle = kf_shuffle.raw ();

		if (kf_train)
			delete kf_train;

		kf_train = kf_data->Subset (Ntrain, shuffle);

		if (kf_test)
			delete kf_test;

		kf_test = kf_data->Subset (Ntest, shuffle + Ntrain);
	}

	DataSet_t *fetchTrainingData (void)
	{
		return kf_train;
	}

	DataSet_t *fetchTestData (void)
	{
		return kf_test;
	}

	void ValidateConfM (NNet_t *mp, const int nFolds, confusion_t &Cm)
	{
		for (int i = 0; i < nFolds; ++i)
		{
			foldData ();

			mp->TrainAndReset (kf_train);

			Cm.Update (kf_test, mp);
		}
	}

	void Validate (NNet_t *mp, const int nFolds, float &mean, float &variance)
	{
		mean = variance = 0.0;

		for (int i = 0; i < nFolds; ++i)
		{
			foldData ();

			mp->TrainAndReset (kf_train);

			IEEE_t guess;
			IEEE_t answer;

			int correct = 0;
			int Ntest = kf_test->N ();

			for (int j = 0; j < Ntest; ++j)
			{
				guess = mp->Compute ((*kf_test)[j]);
 				answer = kf_test->Answer (j);

				if (answer == guess)
					++correct;
			}

			mean += (float) correct / Ntest;
		}

		mean /= nFolds;
	}
};

#endif // header inclusion

