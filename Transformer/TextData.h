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

#ifndef __RANT_TEXTDATA__H__
#define __RANT_TEXTDATA__H__

#include <math.h>

#include <utility>
#include <list>

#include <vocab.h>
#include <lexemes.h>
#include <loss.h>
#include <sampling.h>

#define TOKENWINDOW			100

typedef std::pair<Md_t, int *> exemplar_t;

struct TrainingData_t
{
	TextDocument_t					cd_text;
	vocabDict_t						cd_V;

public:

	TrainingData_t (char const * const textFile,
					char const * const dictFile) :
		cd_text (textFile),
		cd_V (dictFile)
	{
	}

	int get_d (void) const
	{
		return cd_V.getVecDim ();
	}

	int getV_N (void) const
	{
		return cd_V.getVocabN ();
	}
};

class CausalData_t : public TrainingData_t
{
	struct iovec 					ca_lexemes[TOKENWINDOW];

	Md_t							ca_X;
	int								ca_y[TOKENWINDOW];
	exemplar_t						ca_datum;

public:

	CausalData_t (char const * const textFile, char const * const dictFile) :
		TrainingData_t (textFile, dictFile)
	{
	}

	int nextClause ()
	{
		int ca_Ntokens = cd_text.NextClause (TOKENWINDOW, ca_lexemes);

		return ca_Ntokens;
	}

	exemplar_t &getDatum (void)
	{
		/*
		 * We ignore the newline token, hence the Ntokens - 1
		 *
		 */
		int Ntokens = nextClause ();
		if (Ntokens < 0)
		{
			ca_datum.second = NULL;
			return ca_datum;
		}

		ca_X = Md_t (Ntokens - 1, cd_V.getVecDim ());
        int Nvectors = cd_V.LookupTokens (Ntokens - 1, ca_lexemes, ca_X);

		assert (Nvectors == (Ntokens - 1));

        /*
         * Build the causal ground truth
         *
         */
        for (int i = 1; i < Ntokens; ++i)
            ca_y[i - 1] = cd_V[(char const *) ca_lexemes[i].iov_base];

		ca_datum.first = ca_X;
		ca_datum.second = ca_y;

		return ca_datum;
	}

	void reset (void)
	{
		cd_text.Reset ();
	}

};

class MaskedData_t : public TrainingData_t
{
	struct iovec 					be_lexemes[TOKENWINDOW];

	Md_t							be_X;
	int								*be_y;
	exemplar_t						be_datum;
	bool							be_useList;
	typedef std::list<exemplar_t *>	exemplarList_t;
	exemplarList_t					be_dataList;
	exemplarList_t::iterator		be_dataCursor;

	int								be_minLen;
	int								be_maxLen;

public:

	MaskedData_t (char const * const textFile, char const * const dictFile) :
		TrainingData_t (textFile, dictFile),
		be_y (NULL),
		be_useList (false),
		be_minLen (-1),
		be_maxLen (INT_MAX)
	{
	}

	void setMinLen (const int min)
	{
		be_minLen = min;
	}

	void setMaxLen (const int max)
	{
		be_maxLen = max;
	}

	int nextClause ()
	{
		int be_Ntokens = cd_text.NextClause (TOKENWINDOW, be_lexemes);

		return be_Ntokens;
	}

	void maskDatum (const int Ntokens)
	{
        /*
         * Build the masked ground truth
         *
         */

		int numberToMask = (int) ceil (0.2 * (IEEE_t) Ntokens);

		be_y = new int [Ntokens];

		NoReplacementSamples_t toMask (Ntokens);

		for (int ii = 0; ii < Ntokens; ++ii)
		{
			int mask = toMask.Sample ();

			if (numberToMask)
			{

				be_y[mask] = cd_V[(char const *) be_lexemes[mask].iov_base];
				be_lexemes[mask].iov_base = (void *) "qzmUNKqzm";
				be_lexemes[mask].iov_len = 9;

				--numberToMask;

			} else
				be_y[mask] = __MASKED_SKIP_POSITION;

// printf ("%d\t%f\t%s\n", i, sample, (char const *) be_lexemes[i].iov_base);
		}
	}

	exemplar_t &getDatumAnteList (void)
	{
		/*
		 * We ignore the newline token, hence the Ntokens - 1
		 *
		 */
		int Ntokens;

		while (true)
		{
			Ntokens = nextClause ();
			if (Ntokens > be_minLen && Ntokens < be_maxLen)
				break;
		}

		maskDatum (Ntokens);

		be_X = Md_t (Ntokens - 1, cd_V.getVecDim ());
        int Nvectors = cd_V.LookupTokens (Ntokens - 1, be_lexemes, be_X);

		assert (Nvectors == (Ntokens - 1));

		be_datum.first = be_X;
		be_datum.second = be_y;

		exemplar_t *p = new exemplar_t;
		p->first = be_X;
		p->second = be_y;
		be_dataList.push_back (p);

		return be_datum;
	}

	exemplar_t &getDatumList (void)
	{
		exemplar_t *p = *be_dataCursor;
		be_datum = *p;

		++be_dataCursor;

		return be_datum;
	}

	exemplar_t &getDatum (void)
	{
		if (be_useList)
			return getDatumList ();
		else
			return getDatumAnteList ();
	}

	void reset (void)
	{
		cd_text.Reset ();

		be_dataCursor = be_dataList.begin ();
		be_useList = true;
	}

};


#endif // header inclusion

