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

#ifndef __DataSetN_H__
#define __DataSetN_H__

#include <float.h>
#include <math.h>
#include <assert.h>

typedef double * TrainingRow_t;

struct DataSet_t
{
	int						t_N;
	int						t_Nin;
	int						t_Nout;
	TrainingRow_t			t_data;

	DataSet_t (int N, int Nin, int Nout) :
		t_N (N),
		t_Nin (Nin),
		t_Nout (Nout),
		t_data (new double [N * Nin + N * Nout])
	{
		assert (t_Nout == 1);
	}

	DataSet_t (int N, int Nin, int Nout, double *datap) :
		t_N (N),
		t_Nin (Nin),
		t_Nout (Nout),
		t_data (datap)
	{
	}

	~DataSet_t (void)
	{
		delete [] t_data;
	}

	TrainingRow_t entry (const int index) const
	{
		return t_data + index * (t_Nout + t_Nin);
	}

	TrainingRow_t operator[] (int index) const
	{
		return entry (index);
	}

	DataSet_t *Subset (int N, int indices [])
	{
		DataSet_t *p = new DataSet_t (N, t_Nin, t_Nout);
		size_t Nentries = t_Nin + t_Nout;
		size_t len = sizeof (double) * Nentries;
		TrainingRow_t from, to;

		to = p->Raw ();

		for (int i = 0; i < N; ++i)
		{
			from = entry (indices[i]);
			memcpy (to, from, len);
			to += Nentries;
		}

		return p;
	}

	double * Raw (void)
	{
		return t_data;
	}

	double Answer (const int index) const
	{
		return *(t_data + index * (t_Nout + t_Nin) + t_Nin);
	}

	int N (void)
	{
		return t_N;
	}

	void FeatureIteration (int feature, int &stride, double *&base)
	{
		stride = t_Nin + t_Nout; // the data are stored row order
		base = t_data + feature;
	}

	double Max (const int feature)
	{
		double best = -DBL_MAX;
		int stride;
		double *column;

		FeatureIteration (feature, stride, column);

		for (int i = 0; i < t_N; ++i, column += stride)
			if (best < *column)
				best = *column;

		return best;
	}

	double Min (const int feature)
	{
		double best = DBL_MAX;
		int stride;
		double *column;

		FeatureIteration (feature, stride, column);

		for (int i = 0; i < t_N; ++i, column += stride)
			if (best > *column)
				best = *column;

		return best;
	}

	double Mean (const int feature)
	{
		double sum = 0;
		int stride;
		double *column;

		FeatureIteration (feature, stride, column);

		for (int i = 0; i < t_N; ++i, column += stride)
			sum += *column;

		return sum / t_N;
	}

	double Variance (const int feature)
	{
		double sum = 0;
		double sumsq = 0;
		double var;
		int stride;
		double *column;

		FeatureIteration (feature, stride, column);

		for (int i = 0; i < t_N; ++i, column += stride)
		{
			sum += *column;
			sumsq += *column * *column;
		}

		sum /= t_N;
		sum *= sum;

		var = sumsq / (t_N - 1) - sum;

		return var;
	}

	void Center (const int feature)
	{
		double centre = Mean (feature);
		int stride;
		double *column;

		FeatureIteration (feature, stride, column);

		for (int i = 0; i < t_N; ++i, column += stride)
			*column -= centre;
	}

	void Zscore (const int feature, bool centre = true)
	{
		double stddev = sqrt (Variance (feature));
		double mean = (centre ? Mean (feature) : nan (NULL));
		int stride;
		double *column;

		FeatureIteration (feature, stride, column);

		for (int i = 0; i < t_N; ++i, column += stride)
		{
			if (centre)
				*column -= mean;
			*column /= stddev;
		}
	}
};

#endif // header inclusion

