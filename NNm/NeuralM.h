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

#ifndef __DJS_NeuralMatrix__H__
#define __DJS_NeuralMatrix__H__

#include <stdio.h>
#include <string.h>

#include <ANT.h>

/*
 * This file implements the neural matrix.  They implement some
 * matrix/vector products that neural networks need.  They look
 * as follows, a row for each perceptron, and a column for each
 * weight.  The first column contains bias weights.  They are
 * implicit so NeuralM multiplies matrices and vectors and adds
 * the first column.  The matrices are stored row order, that is,
 * in memory they look like { row1, row2, ..., rowN }.
 *
 * There are 2 matrix products, Ax + bias --> y and transpose (A)x --> y
 * They are implemented differently to be cacheline friendly, that is,
 * we try to load each cacheline only once.
 *
 */

inline IEEE_t DotProduct (const int N, IEEE_t * __restrict x, IEEE_t * __restrict y)
{
	IEEE_t dot = 0;
	for (int i = 0; i < N; ++i)
		dot += *x++ * *y++;

	return dot;
}

struct NeuralM_t
{
	bool		sm_releaseMemory;
	int			sm_rows;
	int			sm_columns;
	int			sm_len;
	IEEE_t		*sm_data;

	NeuralM_t (void) :
		sm_releaseMemory (false),
		sm_rows (-1),
		sm_columns (-1),
		sm_len (-1),
		sm_data (NULL)
	{
	}

	NeuralM_t (const int rows, const int columns) :
		sm_releaseMemory (true),
		sm_rows (rows),
		sm_columns (columns),
		sm_len (rows * columns),
		sm_data (new IEEE_t [sm_len])
	{
	}

	NeuralM_t (const int rows, const int columns, IEEE_t *datap) :
		sm_releaseMemory (false),
		sm_rows (rows),
		sm_columns (columns),
		sm_len (rows * columns),
		sm_data (datap)
	{
	}

	~NeuralM_t (void)
	{
		if (sm_releaseMemory && sm_data)
			delete [] sm_data;
	}

	void newVector (int N)
	{
		assert (sm_data == NULL);

		sm_releaseMemory = true;
		sm_rows = N;
		sm_columns = 1;
		sm_len = N;
		sm_data = new IEEE_t [N];
	}

	bool Copy (void)
	{
		if (sm_releaseMemory || sm_data == NULL)
			return false;

		IEEE_t *datap = new IEEE_t [N ()];
		memcpy (datap, sm_data, sizeof (IEEE_t) * sm_len);
		sm_data = datap;
		sm_releaseMemory = true;

		return true;
	}

	bool Copy (NeuralM_t &X)
	{
		assert (X.sm_rows == sm_rows);
		assert (X.sm_columns == sm_columns);

		memcpy (sm_data, X.sm_data, sizeof (IEEE_t) * sm_len);

		return true;
	}

	void Accept (IEEE_t *Xi)
	{
		memcpy (sm_data, Xi, sizeof (IEEE_t) * sm_len);
	}

	IEEE_t *raw (void)
	{
		return sm_data;
	}

	int N (void) const
	{
		return sm_len;
	}

	int rows (void) const
	{
		return sm_rows;
	}

	int columns (void) const
	{
		return sm_columns;
	}

	int stride (void) const
	{
		return columns ();
	}

	// Introduced for vectors
	inline IEEE_t &operator[] (const int row)
	{
		return sm_data[row];
	}

	IEEE_t &operator() (const int row, const int column)
	{
		return *(sm_data + (row * sm_columns) + column);
	}

	IEEE_t &Mentry (const int row, const int column)
	{
		return *(sm_data + (row * sm_columns) + column);
	}

	/*
	 * Assumes first column is the bias (1), so used the submatrix
	 * multiply the vector.
	 *
	 */
	inline
	void MatrixVectorMultBias (NeuralM_t &A, IEEE_t *x)
	{
		IEEE_t *pA = A.sm_data;
		int jump = A.stride ();
		int Nweights = A.columns () - 1;

		for (int i = 0; i < sm_rows; ++i)
		{
			// add the bias to the scaler product, avoid reload of cache line
			sm_data[i] = *pA;
			sm_data[i] += DotProduct (Nweights, pA + 1, x);
			pA += jump;
		}
	}

	/*
	 * Normal matrix multiplication.  No assumption of first column.
	 *
	 * Applies RELU activation.
	 *
	 */
	inline
	void MatrixVectorMultCNN (NeuralM_t &A, IEEE_t *x)
	{
		IEEE_t *pA = A.sm_data;
		int jump = A.stride ();
		int Nweights = A.columns ();

		for (int i = 0; i < sm_rows; ++i)
		{
			// called with zero'ed memory.

			sm_data[i] = RELU (DotProduct (Nweights, pA, x));
			pA += jump;
		}
	}

	/*
	 * Normal matrix multiplication.  No assumption of first column.
	 *
	 */
	inline
	void MatrixVectorMultNoBias (NeuralM_t &A, IEEE_t *x)
	{
		IEEE_t *pA = A.sm_data;
		int jump = A.stride ();
		int Nweights = A.columns ();

		for (int i = 0; i < sm_rows; ++i)
		{
			// called with zero'ed memory.

			sm_data[i] = DotProduct (Nweights, pA, x);
			pA += jump;
		}
	}

	/*
	 * Assumes first column is the bias, and ignores.
	 *
	 */
	inline
	void TransposeMatrixVectorMult (NeuralM_t &A, IEEE_t *vec)
	{
		IEEE_t *rowp = A.sm_data + 1; // skip bias
		int Nweights = A.columns () - 1;
		int runs = A.rows ();
		int jump = A.stride ();

		zero ();

		for (int i = 0; i < runs; ++i)
		{
			for (int j = 0; j < Nweights; ++j)
				sm_data[j] += rowp[j] * vec[i];

			rowp += jump;
		}
	}

	void zero (void)
	{
		memset (sm_data, 0, sm_len * sizeof (IEEE_t));
	}

	void setValue (IEEE_t x)
	{
		IEEE_t *p = raw ();
		int halt = N ();

		for (int i = 0; i < halt; ++i)
			*p++ = x;
	}

	void resize (const int rows, const int columns)
	{
		if (sm_data && sm_releaseMemory)
			delete [] sm_data;

		sm_releaseMemory = true;
		sm_rows = rows;
		sm_columns = columns;
		sm_len = rows * columns;

		sm_data = new IEEE_t [sm_len];

	}

	int Load (FILE *fp, const int rows, const int columns)
	{
		resize (rows, columns);
		IEEE_t *p = raw ();
		int rc;

		for (int i = 0; i < sm_rows; ++i)
		{
			for (int j = 0; j < sm_columns; ++j)
			{
				rc = fscanf (fp, "%le,", p);

				if (rc != 1)
					return -1;

				++p;
			}

			(void) fscanf (fp, "\n");
		}

		return 0;
	}

	void displayMeta (const char * const msgp = NULL, FILE *fp=stdout) const
	{
		int bytes;

		if (msgp)
			bytes = fprintf (fp, "%s\n", msgp);

		bytes = fprintf (fp, "@Meta %d, %d\n", sm_rows, sm_columns);
		if (bytes < 1)
			throw ("WRITE FAILED 1");
	}

	void displayExp (const char * const msgp = NULL, FILE *fp=stdout) const
	{
		int bytes;

		if (msgp)
			fprintf (fp, "%s\n", msgp);

		for (int i = 0, index = 0; i < sm_rows; ++i)
		{
			for (int j = 0; j < sm_columns; ++j, ++index)
				bytes = fprintf (fp, "%.32f%s ",
					sm_data[index],
					(j + 1 != sm_columns ? "," : ""));

			if (bytes < 1)
				throw ("WRITE FAILED 2");

			fprintf (fp, "\n");
		}
	}

	void display (const char * const msgp = NULL, FILE *fp = stdout) const
	{
		if (msgp)
			printf ("%s\n", msgp);

		for (int i = 0, index = 0; i < sm_rows; ++i)
		{
			for (int j = 0; j < sm_columns; ++j, ++index)
				fprintf (fp, "%f%s ",
					sm_data[index],
					(j + 1 != sm_columns ? "," : ""));

			printf ("\n");
		}
	}

	void displayInt (const char * const msgp = NULL) const
	{
		if (msgp)
			printf ("%s\n", msgp);

		for (int i = 0, index = 0; i < sm_rows; ++i)
		{
			for (int j = 0; j < sm_columns; ++j, ++index)
				printf ("%d%s\t",
					(int) sm_data[index],
					(j + 1 != sm_columns ? "," : ""));

			printf ("\n");
		}
	}
};

#endif // header inclusion

