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

#include <string.h>

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

double DotProduct (const int N, double *x, double *y)
{
	double dot = 0;
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
	double		*sm_data;

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
		sm_data (new double [sm_len])
	{
	}

	NeuralM_t (const int rows, const int columns, double *datap) :
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

	double *raw (void)
	{
		return sm_data;
	}

	int N (void)
	{
		return sm_rows * sm_columns;
	}

	int rows (void)
	{
		return sm_rows;
	}

	int columns (void)
	{
		return sm_columns;
	}

	int stride (void)
	{
		return columns ();
	}

	/*
	 * Assumes first column is the bias (1), so used the submatrix
	 * multiply the vector.
	 *
	 */
	void MatrixVectorMult (NeuralM_t &A, double *x)
	{
		double *pA = A.sm_data;
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
	 * Assumes first column is the bias, and ignores.
	 *
	 */
	void TransposeMatrixVectorMult (NeuralM_t &A, double *vec)
	{
		double *rowp = A.sm_data + 1; // skip bias
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
		memset (sm_data, 0, sm_len * sizeof (double));
	}

	void display (const char * const msgp = NULL) const
	{
		if (msgp)
			printf ("%s\n", msgp);

		for (int i = 0, index = 0; i < sm_rows; ++i)
		{
			for (int j = 0; j < sm_columns; ++j, ++index)
				printf ("%e, ", sm_data[index]);
			printf ("\n");
		}
	}
};

#endif // header inclusion

