
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

#ifndef __RANT_VOCAB__H__
#define __RANT_VOCAB__H__

#include <sys/uio.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include <transformer_common.h>
#include <SparseOptimizer.h>

#define __RANT_DICT_ENTRY_LEN			64		// len of lexicon entry
#define IDX2VocabEntry(X) (X * __RANT_DICT_ENTRY_LEN)

#define SEMVECSIZE (vd_d * sizeof (IEEE_t))
#define IDX2OFF(X) (X * SEMVECSIZE)

class vocabDict_t
{
	int						vd_N;
	int						vd_d;		// dimension of semantic vector

	char					*vd_dap;
	IEEE_t					*vd_semanticVectors;

	int dictionaryLookup (char const * const p, bool find = true) const;

	bool					vd_learnV;
	IEEE_t					*vd_dX;
	SparseOptimizer_t		*vd_O;

public:

	vocabDict_t (char const * const filename, const bool learnV=false) :
		vd_learnV (learnV),
		vd_dX (NULL),
		vd_O (NULL)
	{
		FILE *fp;

		fp = fopen (filename, "r");
		if (fp == NULL)
			throw strerror (errno);

		Load (fp);
	}

	void Load (FILE *fp)
	{
		char buffer[128];
		int rc;

		rc = fscanf (fp, "%s %d\n", buffer, &vd_N);
		if (rc != 2)
			throw ("Bad Vocab File");
		if (strcmp (buffer, "Words") != 0)
			throw ("Bad Dictionary");

		rc = fscanf (fp, "%s %d\n", buffer, &vd_d);
		if (rc != 2)
			throw ("Bad Vector Dim");
		if (strcmp (buffer, "Tuple") != 0)
			throw ("Bad Semantics");

		rc = fscanf (fp, "%s\n", buffer);
		if (rc != 1)
			throw ("Bad Dictionary Contents");
		if (strcmp (buffer, "Dictionary") != 0)
			throw ("Bad Semantic Space");

		vd_dap = new char [vd_N * __RANT_DICT_ENTRY_LEN];
		vd_semanticVectors = new IEEE_t [vd_N * vd_d];

		char *p = vd_dap;
		IEEE_t *q = vd_semanticVectors;

		for (int i = 0; i < vd_N; ++i, p += __RANT_DICT_ENTRY_LEN)
		{
			rc = fscanf (fp, "%s\n", p);
			if (rc != 1)
				throw ("Bad Entry");

			for (int j = 0; j < vd_d; ++j, ++q)
				rc = fscanf (fp, "%lf\n", q);
		}

		if (vd_learnV)
		{
			vd_dX = new IEEE_t [vd_N * vd_d];
			vd_O = new SparseOptimizer_t (vd_N, vd_d, vd_semanticVectors, vd_dX);
		}
	}

	~vocabDict_t (void)
	{
		delete [] vd_semanticVectors;
		delete [] vd_dap;

		if (vd_dX)
			delete [] vd_dX;
	}

	int getVocabN (void) const
	{
		return vd_N;
	}

	int getVecDim (void) const
	{
		return vd_d;
	}

	/*
	 * return the tokenID, which corresponds to position in
	 * the dictionary.
	 *
	 */
	int operator[] (char const * const wordp) const
	{
		int index = dictionaryLookup (wordp);

		// printf ("%d\t%s\n", index, vd_dap + IDX2VocabEntry (index));

		return index;
	}

	char const *TokenToString (const int tokenId)
	{
		return vd_dap + tokenId * __RANT_DICT_ENTRY_LEN;
	}

	int EmbedTokens (const int N, struct iovec *pTokens, Md_t &X)
	{
		int valid = 0;
		int index;

		for (int i = 0; i < N; ++i)
		{
			index = (*this)[(char const * const) pTokens[i].iov_base];
			if (index == -1)
			{
				pTokens[i].iov_len = -1;

#if 1
// Currently there is one text.  V and E were built for it.
// This is a temporary dev measure to verify correctness.
printf ("Not found %d\t%s\n", 
	(int) pTokens[i].iov_len, 
	(char *) pTokens[i].iov_base);
#endif

				continue;
			}

			X.importRow (i, vd_semanticVectors + (index * vd_d));
			++valid;
		}

		return valid;
	}

	/*
	 * Used for learnable embeddings.  fit () needs to access the
	 * the initial embeddings.
	 *
	 * Model needs to account for structure: |V| x d, row order.
	 *
	 */
	void backward (Md_t &dX, int const * const y)
	{
		if (vd_learnV == false)
			return;

		// The emebeddings are row order so convert here
		int Ntokens = dX.rows ();
		int token;

		for (int i = 0; i < Ntokens; ++i)
		{
			token = y[i];
			if (token < 0 || token >= vd_N)
				continue;

			vd_O->touch (token);
			dX.exportRow (token, vd_dX + (token * vd_d));
		}
	}

	void update (void)
	{
		if (vd_learnV == false)
			return;

		vd_O->update ();
	}

	int N_LearnableParameters (void) const
	{
		if (vd_learnV)
			return vd_N * vd_d;

		return 0;
	}

	void DumpTokens (void) const
	{
		char const * p = vd_dap;

		for (int i = 0; i < vd_N; ++i, p += __RANT_DICT_ENTRY_LEN)
			printf ("%d\t%s\n", i, p);
	}
};

int vocabDict_t::dictionaryLookup (char const * const p, bool find) const
{
	int left = 0;
	int right = vd_N - 1;
    int cursor;
	int lexCmp;
	bool failed = true;

    while (true) 
    {
        if (left > right) 
        {
            if (right >= 0 && lexCmp < 0)
                cursor = right;
            else 
                cursor = left;

            break; /* while(1) search loop */
        }

        cursor = (right + left) >> 1;

		lexCmp = strcmp (p, vd_dap + cursor * __RANT_DICT_ENTRY_LEN);

        if (lexCmp == 0) 
		{
			failed = false;

            break; /* while(1) search loop */
		}

        if (lexCmp < 0)
            right = cursor - 1;
        else
            left = cursor + 1;
    }

    assert (cursor >= 0 && cursor <= vd_N);

	if (find && failed)
		cursor = -1; // not looking for insert point

    return cursor;
}

#endif // header inclusion


