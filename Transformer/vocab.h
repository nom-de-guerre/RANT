
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

#include <list>
typedef std::list<int> tokenList_t;

#include <transformer_common.h>
#include <SparseOptimizer.h>

#define __RANT_DICT_ENTRY_LEN			64		// len of lexicon entry
#define IDX2VocabEntry(X) (X * __RANT_DICT_ENTRY_LEN)

#define SEMVECSIZE (vd_d * sizeof (IEEE_t))
#define IDX2OFF(X) (X * SEMVECSIZE)

#define __MAXSEQ_LENGTH 100

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

	Md_t					vd_positional;

public:

	vocabDict_t (char const * const filename, const bool learnV=true) :
		vd_learnV (learnV),
		vd_dX (NULL),
		vd_O (NULL)
	{
		FILE *fp;

		fp = fopen (filename, "r");
		if (fp == NULL)
			throw strerror (errno);

		Load (fp);

		buildPositional ();
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
		double entry_buffer;

		for (int i = 0; i < vd_N; ++i, p += __RANT_DICT_ENTRY_LEN)
		{
			rc = fscanf (fp, "%s\n", p);
			if (rc != 1)
				throw ("Bad Entry");

			for (int j = 0; j < vd_d; ++j, ++q)
			{
				rc = fscanf (fp, "%lf\n", &entry_buffer);
				*q = (IEEE_t) entry_buffer;
			}
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

	void printTokens (int N, int const *tokens)
	{
		for (int i = 0; i < N; ++i)
			printf ("%s ", TokenToString (tokens[i]));
		printf ("\n");
	}

	void buildPositional (void)
	{
		IEEE_t d = vd_d;

		vd_positional = Md_t (__MAXSEQ_LENGTH, vd_d);
		vd_positional.setRowOrder ();

		IEEE_t *p = vd_positional.raw ();

		for (IEEE_t pos = 0; pos < __MAXSEQ_LENGTH; ++pos)
			for (int i = 0; i < d; i += 2)
			{
				IEEE_t angle = pos / pow (10000, (IEEE_t) i / d);

				*p = sin (angle);
				++p;
				*p = cos (angle);
				++p;
			}
	}

	int EmbedTokens (const int N, struct iovec *pTokens, Md_t &X, bool pos=false)
	{
		int token;
		tokenList_t seq;

		for (int i = 0; i < N; ++i)
		{
			token = (*this)[(char const * const) pTokens[i].iov_base];
			seq.push_back (token);
		}

		TokensToX (seq, X, pos);

		return seq.size ();
	}

	bool StringToX (char *pString, Md_t &X, bool pos=false) const
	{
		tokenList_t seq;
		char *start = pString;
		int Ntokens = 1; // account for last push following loop
		int token;

		for (char *p = pString; *p; ++p)
		{
			if (isalnum (*p) || *p == '[' || *p == ']')
				continue;

			*p = 0;
			++p;
			token = (*this)[start];
			seq.push_back (token);
			start = p;
			++Ntokens;
		}

		token = (*this)[start];
		seq.push_back (token);

		X = Md_t (Ntokens, getVecDim ());

		return TokensToX (seq, X, pos);
	}

	bool TokensToX (tokenList_t seq, Md_t &X, bool pos) const
	{
		// greatly speeds up computation of Q, K and V in the heads.
		X.setRowOrder ();
		int token;

		int row = 0;
		for (auto it = seq.begin (); it != seq.end (); ++it, ++row)
		{
			token = *it;

			X.importRow (row, vd_semanticVectors + (token * vd_d));
		}

		if (pos)
		{
			Md_t positional = vd_positional.view (0,
											0,
											X.rows (),
											X.columns ());
			X += positional;
		}

		return true;
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
			// masking models hide tokens by making them negative
			if (token < 0)
				token = -token;

			vd_O->touch (token);
			// export from the dX i-th row to token row in the embeddings
			dX.exportAddRow (i, vd_dX + (token * vd_d));
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

	bool save (char const * const filename)
	{
		FILE *fp = fopen (filename, "w");
		if (fp == NULL)
			return false;

		fprintf (fp, "Words %d\n", vd_N);
		fprintf (fp, "Tuple %d\n", vd_d);
		fprintf (fp, "Dictionary\n");

		for (int i = 0, index = 0; i < vd_N; ++i)
		{
			fprintf (fp, "%s\n", vd_dap + (i * __RANT_DICT_ENTRY_LEN));
			for (int j = 0; j < vd_d; ++j, ++index)
				fprintf (fp, "%lf\n", vd_semanticVectors[index]);
		}

		fclose (fp);

		return true;
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


