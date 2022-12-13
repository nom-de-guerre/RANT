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
#include <string.h>
#include <math.h>
#include <assert.h>

#include <ANT.h>

/*
 * The code below is used for computing the classes for classification
 * from non-numeric labels.
 *
 */

#include <map>
struct cmpStr
{
	bool operator()(const char *x, const char *y) const
	{
		return strcmp (x, y) < 0;
	}
};

typedef std::map<const char *, int, cmpStr> unique_t;

struct Categories_t
{
	unique_t			cc_unique;
	int					cc_code;

	Categories_t (void) :
		cc_code (0)
	{
	}

	~Categories_t (void)
	{
	}

	int N (void) const
	{
		return cc_code;
	}

	int Encode (const char * category)
	{
		int classID;

		auto rc = cc_unique.find (category);
		if (rc == cc_unique.end ())
		{
			cc_unique[strdup (category)] = cc_code;

			classID = cc_code;

			++cc_code;

		} else
			classID = rc->second;


		return classID;
	}
};

struct ClassDict_t
{
	struct dictVal_t
	{
		const char	*className;
		int			classID;
	};

	int			cd_N;
	dictVal_t	*cd_dict;

	ClassDict_t (Categories_t &dict) :
		cd_N (dict.N ()),
		cd_dict (new dictVal_t [cd_N])
	{
		for (const auto& [key, value] : dict.cc_unique)
		{
			cd_dict[value].classID = value;
			cd_dict[value].className = key;
		}
	}

	~ClassDict_t (void)
	{
		delete [] cd_dict;
	}

	const char *Name (const int index)
	{
		return cd_dict[index].className;
	}
};

ClassDict_t *
ComputeClasses (
	const int N,
	const int Nfeatures,
	const char *Table,
	IEEE_t *&Tset,
	const int stride)
{
	Categories_t dict;
	int classID;
	const char *p = Table + (Nfeatures - 1) * sizeof (IEEE_t);
	IEEE_t *csv;

	Tset = new IEEE_t [N * Nfeatures];

	for (int i = 0, index = 0; i < N; ++i)
	{
		csv = (IEEE_t *) Table;
		for (int j = 0; j < Nfeatures - 1; ++j, ++index)
			Tset[index] = csv[j];

		classID = dict.Encode (p);

		Tset[index] = classID;

		Table += stride;
		p += stride;
		++index;
	}

	ClassDict_t *dictp = new ClassDict_t (dict);

	return dictp;
}

typedef IEEE_t * TrainingRow_t;

enum types_e { IGNORE, IEEE, CATEGORICAL };

struct DataSet_t
{
	int						t_N;
	int						t_Nin;
	int						t_Nout;
	int						t_columns; // length of a row

	TrainingRow_t			t_data;

	ClassDict_t				*t_dictp;

	types_e					*t_schema;

	DataSet_t (int N, int Nin, int Nout) :
		t_N (N),
		t_Nin (Nin),
		t_Nout (Nout),
		t_columns (Nin + Nout),
		t_data (new IEEE_t [N * Nin + N * Nout]),
		t_dictp (NULL)
	{
		assert (t_Nout == 1);
	}

	DataSet_t (int N, int Nin, int Nout, IEEE_t *datap) :
		t_N (N),
		t_Nin (Nin),
		t_Nout (Nout),
		t_columns (Nin + Nout),
		t_data (datap),
		t_dictp (NULL)
	{
	}

	DataSet_t (int N, int Nin, int Nout, IEEE_t *datap, ClassDict_t *dictp) :
		t_N (N),
		t_Nin (Nin),
		t_Nout (Nout),
		t_columns (Nin + Nout),
		t_data (datap),
		t_dictp (dictp)
	{
	}

	~DataSet_t (void)
	{
		delete [] t_data;
	}

	DataSet_t *Copy (void) const
	{
		DataSet_t *replica = new DataSet_t (t_N, t_Nin, t_Nout);

		memcpy (replica->t_data, t_data, t_N * t_columns * sizeof (IEEE_t));

		return replica;
	}

	int Nin (void) const
	{
		return t_Nin;
	}

	int Stride (void) const
	{
		return t_Nin + t_Nout;
	}

	void * FeatureBase (const int feature)
	{
		return t_data + feature;
	}

	void Display (void) const
	{
		for (int i = 0, index = 0; i < t_N; ++i)
		{
			for (int j = 0; j < t_columns; ++j, ++index)
				printf ("%f\t", t_data[index]);

			printf ("\n");
		}
	}

	TrainingRow_t entry (const int index) const
	{
		return t_data + index * (t_columns);
	}

	TrainingRow_t operator[] (int index) const
	{
		return entry (index);
	}

	DataSet_t *Subset (int N, int const * const indices) const
	{
		DataSet_t *p = new DataSet_t (N, t_Nin, t_Nout);
		size_t len = sizeof (IEEE_t) * t_columns;
		TrainingRow_t from, to;

		to = p->Raw ();

		for (int i = 0; i < N; ++i)
		{
			from = entry (indices[i]);
			memcpy (to, from, len);
			to += t_columns;
		}

		p->t_dictp = t_dictp;		// needs to be reference counted
		p->t_schema = t_schema;

		return p;
	}

	IEEE_t *Raw (void)
	{
		return t_data;
	}

	IEEE_t Answer (const int index) const
	{
		return *(t_data + index * (t_Nout + t_Nin) + t_Nin);
	}

	IEEE_t * const AnswerVec (const int index) const
	{
		return (t_data + index * (t_Nout + t_Nin) + t_Nin);
	}

	int N (void) const
	{
		return t_N;
	}

	void FeatureIteration (int feature, int &stride, IEEE_t *&base) const
	{
		// the data are stored row order, and elements are assumed IEEE_t
		stride = t_columns;
		base = t_data + feature;
	}

	IEEE_t Max (const int feature) const
	{
		IEEE_t best = -DBL_MAX;
		int stride;
		IEEE_t *column;

		FeatureIteration (feature, stride, column);

		for (int i = 0; i < t_N; ++i, column += stride)
			if (best < *column)
				best = *column;

		return best;
	}

	IEEE_t Min (const int feature) const
	{
		IEEE_t best = DBL_MAX;
		int stride;
		IEEE_t *column;

		FeatureIteration (feature, stride, column);

		for (int i = 0; i < t_N; ++i, column += stride)
			if (best > *column)
				best = *column;

		return best;
	}

	IEEE_t Mean (const int feature) const
	{
		IEEE_t sum = 0;
		int stride;
		IEEE_t *column;

		FeatureIteration (feature, stride, column);

		for (int i = 0; i < t_N; ++i, column += stride)
			sum += *column;

		return sum / t_N;
	}

	IEEE_t Variance (const int feature) const
	{
		IEEE_t mean;
		IEEE_t summand;
		IEEE_t var;
		int stride;
		IEEE_t *column;
		IEEE_t sum = 0;

		/*
		 * We use the double pass version as it is more numerically stable
		 * than the single pass version: E(x^2) - E(x)^2
		 *
		 */
		mean = Mean (feature);

		FeatureIteration (feature, stride, column);

		for (int i = 0; i < t_N; ++i, column += stride)
		{
			summand = *column - mean;
			sum += summand * summand; 
		}

		var = sum / (t_N - 1);

		return var;
	}

	IEEE_t StdDev (const int feature) const
	{
		return sqrt (Variance (feature));
	}

	void Center (const int feature)
	{
		IEEE_t centre = Mean (feature);
		int stride;
		IEEE_t *column;

		FeatureIteration (feature, stride, column);

		for (int i = 0; i < t_N; ++i, column += stride)
			*column -= centre;
	}

	void Zscore (const int feature, bool centre = true)
	{
		IEEE_t stddev = StdDev (feature);
		IEEE_t mean = (centre ? Mean (feature) : nan (""));
		int stride;
		IEEE_t *column;

		FeatureIteration (feature, stride, column);

		for (int i = 0; i < t_N; ++i, column += stride)
		{
			if (centre)
				*column -= mean;
			*column /= stddev;
		}
	}

	void Normalize (const int feature)
	{
		int stride;
		IEEE_t *column;
		IEEE_t sum = 0;

		Center (feature); // mean is now zero

		FeatureIteration (feature, stride, column);

		for (int i = 0; i < t_N; ++i, column += stride)
			sum += *column * *column;

		sum = sqrt (sum / (t_N - 1));

		FeatureIteration (feature, stride, column);

		for (int i = 0; i < t_N; ++i, column += stride)
			*column /= sum;
	}

	void Normalize (void)
	{
		for (int i = 0; i < t_Nin; ++i)
		{
			if (t_schema && t_schema[i] != IEEE)
				continue;

			Normalize (i);
		}
	}

	const char *CategoryName (const int type)
	{
		return t_dictp->Name (type);
	}
};

#endif // header inclusion

