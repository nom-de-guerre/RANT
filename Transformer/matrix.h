/*

Copyright (c) 2015, Douglas Santry
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

#ifndef __DJS_MATRIX_H__
#define __DJS_MATRIX_H__

#include <limits.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <assert.h>

#include <memory>

static int serialNo = 0;

#define MACH_EPS 2.2204460492503131e-16 // for IEEE double (64 bits)
#define SIGN(X) (signbit (X) ? -1 : 1)

/**********************************************************
 *
 * Copy-on-write memory efficient matrices.
 *
 * The C++ types are:
 * Matrix_t -> MatrixView_t -> MatrixData_t
 *
 * Matrix_t used for computation.
 * MatrixView_t maintains a window into matrix data
 * MatrixData_t maintains the raw memory where a matrix lives
 *
 *
 **********************************************************/

/**********************************************************
 *
 * Underlying physcial matrix encapsulated by Matrix_t with pointer
 * (not exposed)
 *
 * The data are laid out in column order, that is, a column is contiguous
 * in memory (and CPU cache line).
 *
 **********************************************************/

template<typename T> struct MatrixData_t
{
	T				*md_data;
	ssize_t			md_rows;
	ssize_t			md_columns;

	MatrixData_t (ssize_t rows, ssize_t columns) :
	md_rows (rows),
	md_columns (columns)
	{
		md_data = new T [md_rows * md_columns];
	}

	~MatrixData_t ()
	{
		delete [] md_data;
	}

	ssize_t rows ()
	{
		return md_rows;
	}

	ssize_t columns ()
	{
		return md_columns;
	}

	T * raw ()
	{
		return md_data;
	}
};

/**********************************************************
 *
 * View into an underlying matrix 
 * (not exposed)
 *
 **********************************************************/

template<typename T> struct MatrixView_t
{
	std::shared_ptr< MatrixData_t<T> > mw_matrix;
	T					*mw_base;

	int					mw_serialNo;
	bool				mw_transpose;
	bool				mw_CoW;
	bool				mw_immutable;

	// The physical size of the matrix
	int					mw_prows;
	int					mw_pcolumns;

	// The virtual size of the matrix (for restrictions and block computations)
	int 				mw_vrows;
	int					mw_vcolumns;

	// are the operations +/- defined?
	bool defined (MatrixView_t &M)
	{
		if (M.rows () == rows () && M.columns () == columns ())
			return true;

		return false;
	}

	static bool defined (MatrixView_t *A, MatrixView_t *B)
	{
		return A->defined (*B);
	}

	// matrix product valid?
	static bool defined_prod (MatrixView_t *A, MatrixView_t *B)
	{
		return A->columns () == B->rows ();
	}

	void init (void)
	{
	}

	MatrixView_t (void)
	{
		assert (false);
	}

	/*
	 * A window into a matrix at <start row, start col> to 
	 * <srow + nrow, scol + ncol>
	 *
	 * Relative to current window.
	 *
	 * CoW
	 *
	 */
	MatrixView_t (MatrixView_t<T> *matrixView,
					int srow,		// start
					int scol,
					int nrows,		// delta from start
					int ncols) :
		mw_matrix (matrixView->mw_matrix),
		mw_serialNo (++serialNo),
		mw_transpose (false),
		mw_CoW (true),
		mw_immutable (false),
		mw_prows (matrixView->mw_prows),
		mw_pcolumns (matrixView->mw_pcolumns),
		mw_vrows (nrows),
		mw_vcolumns (ncols)
	{
		mw_base = matrixView->mw_base + scol * mw_prows + srow;

		init ();
	}

	/*
	 * shallowCoW
	 *
	 */
	MatrixView_t (MatrixView_t<T> *matrixView)
	{
		*this = *matrixView;

		init ();
	}

	/*
	 * A view of a column vector.
	 *
	 * CoW
	 *
	 */
	MatrixView_t (MatrixView_t<T> *matrixView, const int column) :
		mw_matrix (matrixView->mw_matrix),
		mw_serialNo (++serialNo),
		mw_transpose (false),
		mw_CoW (true),
		mw_immutable (false),
		mw_prows (matrixView->rows ()),
		mw_pcolumns (1),
		mw_vrows (mw_prows),
		mw_vcolumns (1)
	{
		mw_base = mw_matrix->raw () + column * matrixView->mw_prows;

		init ();
	}

	/*
	 * A new matrix of dimension __n x __m
	 *
	 * CoW
	 *
	 */
	MatrixView_t (int __n, int __m) :
		mw_matrix (new MatrixData_t<T> (__n, __m)),
		mw_serialNo (++serialNo),
		mw_transpose (false),
		mw_CoW (true),
		mw_immutable (false),
		mw_prows (__n),
		mw_pcolumns (__m),
		mw_vrows (__n),
		mw_vcolumns (__m)
	{
		mw_base = mw_matrix->raw ();

		init ();
	}

	/*
	 * A new matrix of dimension __n x __m, initialized from __A[]
	 * __A[] is in row order.
	 *
	 * CoW
	 *
	 */
	MatrixView_t (int __n, int __m, T __A[]) :
		mw_matrix (new MatrixData_t<T> (__n, __m)),
		mw_serialNo (++serialNo),
		mw_transpose (false),
		mw_CoW (true),
		mw_immutable (false),
		mw_prows (__n),
		mw_pcolumns (__m),
		mw_vrows (__n),
		mw_vcolumns (__m)
	{
		mw_base = mw_matrix->raw ();

		init ();

		int index = 0;
		int phys = 0;

		for (int i = 0; i < mw_vrows; i++)
		{
			phys = i;
			for (int j = 0; j < mw_vcolumns; ++j, ++index, phys += mw_prows)
				mw_base[phys] = __A[index];
		}
	}

	/*
	 * A new matrix of dimension __n x __m.  Initial value of elements
	 * is __x, or a diagonal matrix (e.g. __x = 1 creates I).
	 *
	 * CoW
	 *
	 */
	MatrixView_t (int __n, int __m, T __x, bool not_Diagonal = false) :
		mw_matrix (new MatrixData_t<T> (__n, __m)),
		mw_serialNo (++serialNo),
		mw_transpose (false),
		mw_CoW (true),
		mw_immutable (false),
		mw_prows (__n),
		mw_pcolumns (__m),
		mw_vrows (__n),
		mw_vcolumns (__m)
	{
		mw_base = mw_matrix->raw ();

		init ();

		int phys;

		for (int i = 0; i < mw_prows; i++)
		{
			phys = i;

			for (int j = 0; j < mw_pcolumns; ++j, phys += mw_prows)
			{
				if (!not_Diagonal) {
					if (i == j)
						mw_base[phys] = __x;
					else
						mw_base[phys] = 0;
				} else
					mw_base[phys] = __x;
			}
		}
	}

	~MatrixView_t (void)
	{
	}

	// If we're a window into a sub-matrix explode to full view again
	void viewSelf (void)
	{
		mw_vrows = mw_prows = mw_matrix->md_rows;
		mw_vcolumns = mw_pcolumns = mw_matrix->md_columns;
		mw_base = mw_matrix->raw ();
	}

	void update ()
	{
		mw_serialNo = ++serialNo;
	}

	int rows () 
	{ 
		return mw_vrows; 
	}

	int prows () 
	{ 
		return mw_prows; 
	}

	int columns () 
	{ 
		return mw_vcolumns; 
	}

	int pcolumns () 
	{ 
		return mw_pcolumns; 
	}

	bool immutable ()
	{
		return mw_immutable;
	}

	void set_immutable ()
	{
		mw_immutable = true;
	}

	void set_mutable ()
	{
		mw_immutable = false;
	}

	bool CoW ()
	{
		return mw_CoW; // logical or physical assigment
	}

	void set_WiP (void) // write in place
	{
		mw_CoW = false;
	}

	void set_CoW (void) // copy on write (default)
	{
		mw_CoW = true;
	}

	T * raw ()
	{
		return mw_base;
	}

	void displayMeta (const char *name)
	{
		printf ("%s\t%p (%p)\t%d\t%d (%d) x %d (%d)\t (%s)\n", name,
			this,
			mw_matrix->raw (),
			mw_serialNo,
			rows(),
			mw_prows,
			columns (),
			mw_pcolumns,
			(mw_CoW ? "CoW" : "WiP"));
	}

	void display (const char *name, const char *precision)
	{
		printf ("%s\t%p (%p)\t%d\t%d (%d) x %d (%d)\t (%s)\n", name,
			this, 
			mw_matrix->raw (),
			mw_serialNo, 
			rows(), 
			mw_prows,
			columns (),
			mw_pcolumns,
			(mw_CoW ? "CoW" : "WiP"));

		char buffer [16];
		snprintf (buffer, 16, "\t%%.%sf", precision);

		for (int i = 0; i < rows(); i++)
		{
			for (int j = 0; j < columns(); j++)
				printf (buffer, (double) this->datum (i, j));

			printf ("\n" );
		}
	}

	void displayExp (const char *name)
	{
		printf ("%s\t%p (%p)\t%d\t%d (%d) x %d (%d)\t (%s)\n", name,
			this, 
			mw_matrix->raw (),
			mw_serialNo, 
			rows(), 
			mw_prows,
			columns (),
			mw_pcolumns,
			(mw_CoW ? "CoW" : "WiP"));

		for (int i = 0; i < rows(); i++)
		{
			for (int j = 0; j < columns(); j++)
				printf ("%e\t", this->datum (i, j));

			printf ("\n" );
		}
	}

	inline T &datum (const int row, const int column)
	{
		return mw_base[column * mw_prows + row];
	}

	/*
	 * The follow operators are invoked by Matrix_t; they are not
	 * exposed.
	 *
	 */
	MatrixView_t &operator+= (MatrixView_t &term)
	{
		if (!defined (term))
			throw ("+= illegal dimensions");

		update ();

		T * __restrict base = raw ();
		T * __restrict Abase = term.raw ();
		ssize_t drow = mw_prows - mw_vrows;
		ssize_t Arow = term.mw_prows - term.mw_vrows;

		for (int j = 0; j < mw_vcolumns; 
			++j, base += drow, Abase += Arow)

			for (int i = 0; i < mw_vrows; ++i, ++base, ++Abase)
				*base += *Abase;

		return *this;
	}

	MatrixView_t &operator-= (MatrixView_t &term)
	{
		if (!defined (term))
			throw ("-= illegal dimensions");

		update ();

		T * __restrict base = raw ();
		T * __restrict Abase = term.raw ();
		ssize_t drow = mw_prows - mw_vrows;
		ssize_t Arow = term.mw_prows - term.mw_vrows;

		for (int j = 0; j < mw_vcolumns; 
			++j, base += drow, Abase += Arow)

			for (int i = 0; i < mw_vrows; ++i, ++base, ++Abase)
				*base -= *Abase;

		return *this;
	}

	MatrixView_t &operator*= (T scaler)
	{
		update ();

		T * __restrict base = raw ();
		ssize_t drow = mw_prows - mw_vrows;

		for (int j = 0; j < mw_vcolumns; ++j, base += drow)
			for (int i = 0; i < mw_vrows; ++i, ++base)
				*base *= scaler;

		return *this;
	}

	MatrixView_t &operator/= (T scaler)
	{
		update ();

		T * __restrict base = raw ();
		ssize_t drow = mw_prows - mw_vrows;

		for (int j = 0; j < mw_vcolumns; ++j, base += drow)
			for (int i = 0; i < mw_vrows; ++i, ++base)
				*base /= scaler;

		return *this;
	}
};

template<typename T> struct mptr_t : public std::shared_ptr< MatrixView_t <T> >
{
	mptr_t (void) :
		std::shared_ptr< MatrixView_t <T> > (NULL)
	{
	}

	mptr_t (MatrixView_t <T> *p) : 
		std::shared_ptr< MatrixView_t <T> > (p)
	{
	}

	~mptr_t (void)
	{
	}

	bool valid (void)
	{
		return ((*this) ? true : false);
	}

	int rcount (void)
	{
		return std::shared_ptr< MatrixView_t <T> >::use_count ();
	}

	bool exclusive (void)
	{
#ifdef __DEBUG
		assert (valid ());
#endif

		return (rcount () == 1 
			? true 
			: false);
	}
};

/**********************************************************
 *
 * Matrix type exposed for use in computation.
 *
 * None of this stuff is thread-safe.
 *
 **********************************************************/

template<typename T> class Matrix_t
{
// typedef std::shared_ptr< MatrixView_t<T> > data_t;
#define INVOKE (m_data.get ())

	// data_t			m_data;
	// std::shared_ptr< MatrixView_t<T> > m_data;
	mptr_t<T>			m_data;

	MatrixView_t<T>	*get ()
	{
		return m_data.get ();
	}

	void CoWShallow (void)
	{
		if (m_data.exclusive ())
			return;

		m_data = new MatrixView_t<T> (m_data.get ());
	}

	// Determine if copy-on-write is neccessary
	inline void CoW (void)
	{
		if (INVOKE->immutable ())
			throw ("protection violation");

		// Should we WiP?
		if (!INVOKE->CoW ())
			return;

		/*
		 * Matrix_t is exclusive as is the matrixView - nothing to do
		 *
		 */
		if (m_data.exclusive ())
			if (INVOKE->mw_matrix.use_count () == 1)
				return;

		mptr_t<T> old = m_data;
		m_data = new MatrixView_t<T> (INVOKE->rows (), INVOKE->columns ());

		copy (old);
	}

	// copy a matrix
	void copy (mptr_t<T> old)
	{
		/*
		 * Copying can take the form of:
		 *
		 * (i) physical to physical
		 * (ii) logical to physical
		 *
		 */

		if (old->rows () == old->prows () &&
			old->columns () == old->pcolumns ())
		{
			ssize_t len = old->rows () * old->columns () * sizeof (T);

			// memcpy is safe - the matrices should NOT overlap
			memcpy (m_data->raw (), old->raw (), len);

			return;
		}

		ssize_t prows = old->prows ();
		ssize_t vrows = old->rows ();
		T * __restrict to_ptr = INVOKE->raw ();
		T * __restrict from_ptr = old->raw ();

		for (int columns = INVOKE->columns (); columns; --columns) {

			// memcpy is safe as the memory does not overlap
			memcpy (to_ptr, from_ptr, vrows * sizeof (T));
			to_ptr += vrows;
			from_ptr += prows;
		}
	}

	friend Matrix_t<T> prepare_target (Matrix_t<T> A, Matrix_t<T> B)
	{
		return Matrix_t<T> (A.rows (), B.columns ());
	}

public:

	/**********************************************************
	 *
	 * Constructors
	 *
	 **********************************************************/

	Matrix_t<T> ()
	{
	}

	Matrix_t<T> (Matrix_t<T> &A) :
		m_data (A.m_data)
	{
	}

	Matrix_t<T> (const Matrix_t<T> &A) :
		m_data (A.m_data)
	{
	}

	Matrix_t<T> (int __m, int __n) :
		m_data (new MatrixView_t<T>(__m, __n))
	{
	}

	// initialize matrix from m x n array __A
	Matrix_t<T> (int __m, int __n, T __A[]) :
		m_data (new MatrixView_t<T>(__m, __n, __A))
	{
	}

	/*
	 * m x n matrix, all elements are set to init, unless every is false, 
	 * in which case only the diagonal is set to init.
	 *
	 * e.g., calling with init = 1, and every = false (default) creates
	 * a m x n identity matrix.
	 *
	 */
	Matrix_t (int __m, int __n, T init, bool every = false) :
		m_data (new MatrixView_t<T>(__m, __n, init, every))
	{
	}

	/**********************************************************
	 *
	 * Assignments and misc.
	 *
	 **********************************************************/

	// if using the raw pointer into a window then the stride is the number
	// of rows (in memory footprint is column order to support vectors)
	int stride (void)
	{
		if (!m_data.valid ())
			return -1;

		return INVOKE->mw_prows;
	}

	Matrix_t<T> &operator= (Matrix_t A)
	{
		if (m_data.get () != NULL && INVOKE->immutable ())
			throw ("illegal assignment");

		m_data = A.m_data;

		return *this;
	}

	// unique identifier of the matrix
	int ID (void)
	{
		int tmp = reinterpret_cast<std::uintptr_t>(m_data.get ());
		return (int) tmp;
	}

	// Logical number of rows
	int rows(void)
	{
		return INVOKE->rows ();
	}

	// Physical number of rows
	int prows(void)
	{
		return INVOKE->prows ();
	}

	int columns(void)
	{
		return INVOKE->columns ();
	}

	void displayMeta (const char *name)
	{
		INVOKE->displayMeta (name);
	}

	void display(const char *name = "", const char *precision = "2")
	{
		INVOKE->display(name, precision);
	}

	void displayExp(const char *name = "")
	{
		INVOKE->displayExp (name);
	}

	T *raw ()
	{
		return m_data->raw ();
	}

	void copy (void)
	{
		CoW ();
	}

	/**********************************************************
	 *
	 * Matrix operations
	 * mutable (so CoW must be called first thing)
	 *
	 **********************************************************/

	void randomly_fill_pos (T max)
	{
		CoW ();

		int rows = INVOKE->rows ();
		int columns = INVOKE->columns ();

		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < columns; ++j)
				(*this)(i, j) = max * (T) ((double) rand () / RAND_MAX);
	}

	void randomly_fill (T max, int inv_pc_neg = 4, unsigned seed = 0)
	{
		CoW ();

		int rows = INVOKE->rows ();
		int columns = INVOKE->columns ();

		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < columns; ++j)
			{
				(*this)(i, j) = max * (T) ((double) rand () / RAND_MAX);
				if ((rand () % inv_pc_neg) == 0)
					(*this)(i, j) = -(*this)(i, j);
			}
	}

	// copy a vector into a column of a matrix
	void vec_fill (Matrix_t &v, int index)
	{
#ifdef __DEBUG
		assert (v.rows () == rows ());
		assert (index >= 0 && index < columns ());
#endif

		CoW ();

		T * __restrict base = raw () + m_data->prows () * index;
		T * __restrict vbase = v.raw ();

		for (int i = m_data->rows (); i; --i)
			*base++ = *vbase++;
	}

	T &operator() (int row, int column)
	{
#ifdef __DEBUG
		assert (row >= 0 && row < this->rows ());
		assert (column >= 0 && column < this->columns ());
#endif

#ifndef __UNSAFE_ASSIGNMENT
		// CoW is `expensive' and it can be turned off at compile time
		CoW ();
#endif
		return INVOKE->datum (row, column);
	}

	Matrix_t &operator+=(Matrix_t A)
	{
		CoW ();

		INVOKE->operator+= (*A.get ());

		return *this;
	}

	Matrix_t &operator-=(Matrix_t A)
	{
		CoW ();

		INVOKE->operator-= (*A.get ());

		return *this;
	}

	Matrix_t &operator/=(T scaler)
	{
		CoW ();

		INVOKE->operator/= (scaler);

		return *this;
	}

	Matrix_t &operator*=(T scaler)
	{
		CoW ();

		INVOKE->operator*= (scaler);

		return *this;
	}

	Matrix_t &operator*=(Matrix_t A)
	{
		CoW ();

		Matrix_t<T> __M = *this * A;
		(*this) = __M;

		return *this;
	}

	T vec_magnitude (void)
	{
#ifdef __DEBUG
		assert (columns () == 1);
#endif

		T u = sqrt (vec_dotT ());

		return u;
	}

	Matrix_t &vec_norm ()
	{
#ifdef __DEBUG
		assert (columns () == 1);
#endif

		CoW ();

		T M;

		M = vec_magnitude ();

		*this /= M;

		return *this;
	}

	void pipe (Matrix_t<T> v)
	{
#ifdef __DEBUG
		assert (v.rows () == rows () && v.columns () == columns ());
#endif
		copy (v.m_data);
	}

	Matrix_t<T> solveQR (Matrix_t<T> &b)
	{
		Matrix_t x;

		find_R (b);
		x = find_x (b);

		return x;
	}

	// QR (Householder) based stuff
	void find_R (Matrix_t<T> &);
	Matrix_t<T> find_x (Matrix_t<T> &);
	void QR (Matrix_t<T> &);
	void HessenbergSimilarity (bool similar = true);
	void ApplyHouseholder (Matrix_t<T> &, bool similar = false);
	void ImplicitQRStep (int, double [], Matrix_t &);
	bool ComputeCholesky (Matrix_t<T> &);
	void SolveLower (Matrix_t<T> &, Matrix_t<T> &);
	void SolveUpper (Matrix_t<T> &);

	/**********************************************************
	 *
	 * Miscellaneous matrix operations
	 *
	 * Idempotent (do not call CoW)
	 *
	 **********************************************************/

	void set_WiP ()
	{
		CoWShallow ();

		INVOKE->set_WiP ();
	}

	void set_CoW ()
	{
		CoWShallow ();

		INVOKE->set_CoW ();
	}

	void set_ro ()
	{
		CoWShallow ();

		INVOKE->set_immutable ();
	}

	void set_rw ()
	{
		CoWShallow ();

		INVOKE->set_mutable ();
	}

	/*
	 * Create a view of the matrix.  WiP true results in assignments to the
	 * view being reflected in the underlying matrix.  The view looks like:
	 * <srow, scol> to <srow + nrows, scol + ncols>
	 *
	 */
	Matrix_t view (int srow, int scol, int nrows, int ncols, bool WiP = true)
	{
		Matrix_t A;

		A.m_data =  new MatrixView_t<T> (m_data.get (),
										srow,
										scol,
										nrows,
										ncols);

		if (WiP)
			A.set_WiP ();

		return A;
	}

	/*
	 * If the logical view is not equal to the physical view (e.g. a
	 * column vector) dilate the view to compass the entire physical
	 * matrix.
	 *
	 */
	void viewOriginal (void)
	{
		INVOKE->viewSelf ();
	}

	/*
	 * Create a view of a matrix starting at <0, 0> to <rows, columns>.
	 * WiP true will write in place.
	 *
	 */
	void viewBlock (int rows, int  columns)
	{
		m_data->mw_vrows = rows;
		m_data->mw_vcolumns = columns;
	}

	/*
	 * Create a vector view in a matrix at column.  Writes in place.
	 *
	 */
	Matrix_t vec_view (const int column, bool WiP = true)
	{
#ifdef __DEBUG
		assert (column >= 0 && column < m_data->pcolumns ());
#endif

		Matrix_t v;
		v.m_data = new MatrixView_t<T> (m_data.get (), column);

		if (WiP)
			v.set_WiP ();

		return v;
	}

	/*
	 * Verify equality of two matrices, +/- epsilon
	 *
	 */
	bool equal_eps (Matrix_t<T> &A, T epsilon)
	{
		if (!MatrixView_t<T>::defined (A.get(), INVOKE))
			throw ("+/- epsilon dimension mismatch");

		int __rows = rows ();
		int __columns = columns ();

		for (int i = 0; i < __rows; ++i)
			for (int j = 0; j < __columns; ++j)
				if ((T) fabs ((double) A(i, j) - (*this)(i, j)) > epsilon)
					return false;

		return true;
	}

	bool operator== (Matrix_t &A)
	{
		ssize_t len = INVOKE->rows () * INVOKE->columns ();

		T * __restrict x = raw ();
		T * __restrict y = A.raw ();

		for (; len; --len)
			if (*x++ != *y++)
				return false;

		return true;
	}

	// <x^t, x> - useful for normalizing vectors
	T vec_dotT ()
	{
		T * __restrict x = raw ();
		int __rows = m_data->rows ();
		T dot = 0;

		for (int i = 0; i < __rows; ++i, ++x)
			dot += *x * *x;

		return dot;
	}

	// dot product of 2 vectors (matrices of the form (rows, 1))
	T vec_dot (Matrix_t &y)
	{
#ifdef __DEBUG
		assert (columns () == y.columns () == 1);
		assert (rows () == y.rows ());
#endif

		int __rows = rows ();
		T * __restrict yraw = y.raw ();
		T * __restrict x = raw ();
		T dot = 0;

		for (int i = 0; i < __rows; ++i, ++x, ++yraw)
			dot += *x * *yraw;

		return dot;
	}

	// returns a new matrix that that is the transpose
	Matrix_t<T> transpose()
	{
		int rows = INVOKE->rows ();
		int prows = INVOKE->prows ();
		int columns = INVOKE->columns ();
		Matrix_t<T> At (columns, rows);

		T * __restrict A;
		T * __restrict p = raw ();
		T * __restrict __A = At.raw ();

		for (int i = 0; i < rows; i++)
		{
			A = p + i;
			for (int j = 0; j < columns; j++)
			{
				*__A = *A;
				A += prows;
				++__A;
			}
		}

		return At;
	}

	// The matrix infinite norm
	double norm_inf (void)
	{
		double max = DBL_MIN;

		for (int i = INVOKE->rows () - 1; i >= 0; --i) {

			double norm = 0;
			for (int j = INVOKE->columns () - 1; j >= 0; --j)
				norm += fabs ((*this)(i, j));

			if (norm > max)
				max = norm;
		}

		return max;
	}

	// The matrix Frobenius norm
	void zero (void)
	{
		T * __restrict p = INVOKE->raw ();
		int rows = INVOKE->rows ();
		int columns = INVOKE->columns ();

		memset (p, 0, rows * columns * sizeof (T));
	}

	int N (void) const
	{
		return INVOKE->rows () * INVOKE->columns ();
	}

	void importRow (int row, T *p)
	{
		T *datap = raw () + row;
		int incr = stride ();
		int d = columns ();

		for (int i = 0; i < d; ++i, datap += incr)
			*datap = p[i];
	}

	void importRow (int row, Matrix_t<T> &A)
	{
		if (!MatrixView_t<T>::defined (get(), A.get()))
			throw ("importRow dimension mismatch");

		T *to = raw () + row;
		T *from = A.raw () + row;

		int incr = stride ();
		int d = columns ();

		for (int i = 0; i < d; ++i, to += incr, from += incr)
			*to = *from;
	}

	// The matrix Frobenius norm
	T Frobenius (void)
	{
		T * __restrict p = INVOKE->raw ();
		int rows = INVOKE->rows ();
		int columns = INVOKE->columns ();
		int delta = INVOKE->pcolumns () - columns;
		T norm = 0;

		for (int i = 0; i < rows; ++i)
		{
			for (int j = 0; j < columns; ++j, ++p)
				norm += *p * *p;

			p += delta;
		}

		return sqrt (norm);
	}

	// For multiplying a matrix by its transpose: B = transpose (A) * A
	friend Matrix_t transpose (Matrix_t<T> A)
	{
		A.m_data->mw_transpose = true;

		return A;
	}

	friend Matrix_t<T> operator*(Matrix_t<T> A, Matrix_t<T> B)
	{
		if (B.m_data->mw_transpose && A.raw () != B.raw ())
			throw ("only AtB or AtA supported"); // need to implement this

		if (A.m_data->mw_transpose) {

			A.m_data->mw_transpose = false;

			return mult_transpose (A, B);
		}

		return mult_normal (A, B);
	}

	friend Matrix_t<T> mult_normal (Matrix_t<T> A, Matrix_t<T> B)
	{
		if (!MatrixView_t<T>::defined_prod (A.get(), B.get()))
			throw ("* dimension mismatch");
 
		int Arows =  A.rows ();
		int Aprows = A.m_data->prows ();
		int Acolumns = A.columns ();
		int Bcolumns = B.columns ();
		int Bdelta = B.m_data->prows () - B.rows ();

		Matrix_t<T> Q = prepare_target (A, B);
		int Qrows = Q.prows ();

		T * __restrict Araw = A.raw ();
		T * __restrict Braw = B.raw ();
		T * __restrict Qraw = Q.raw ();
		T * __restrict Aidx;
		T * __restrict Bidx;
		T * __restrict Qidx;
		int Qinc = 0;

		for (int i = 0; i < Arows; i++)
		{
			Bidx = Braw;
			Qinc = 0;

			for (int j = 0; j < Bcolumns; j++)
			{
				Aidx = Araw + i;
				Qidx = Qraw + Qinc + i;
				*Qidx = 0;

				for (int k = 0; k < Acolumns; k++)
				{
					*Qidx += *Aidx * *Bidx;

					Aidx += Aprows;
					++Bidx;
				}

				Qinc += Qrows;
				Bidx += Bdelta;
			}
		}

		return Q;
	}

	/*
	 * It is logically transposed, not physically
	 *
	 */
	friend Matrix_t<T> mult_transpose (Matrix_t<T> A, Matrix_t<T> B)
	{
		int Arows =  A.rows ();
		int Acolumns = A.columns ();
		int Aprows = A.m_data->prows ();
		int Bcolumns = B.columns ();
		Matrix_t<T> Q (Acolumns, B.columns ());
		int Bdelta = B.m_data->prows () - B.rows ();

		if (Arows != B.rows ())
			throw ("* AtB dimension mismatch");

		T * __restrict Araw = A.raw ();
		T * __restrict Braw = B.raw ();
		T * __restrict Qraw = Q.raw ();
		T * __restrict Aidx;
		T * __restrict Bidx;
		T * __restrict Qidx;
		int Qinc = 0;
		int Ainc = 0;

		for (int i = 0; i < Acolumns; i++)
		{
			Bidx = Braw;
			Qinc = 0;

			for (int j = 0; j < Bcolumns; j++)
			{
				Aidx = Araw + Ainc;
				Qidx = Qraw + Qinc + i;
				*Qidx = 0;

				for (int k = 0; k < Arows; k++)
				{
					*Qidx += *Aidx * *Bidx;

					++Aidx;
					++Bidx;
				}

				Qinc += Acolumns;
				Bidx += Bdelta;
			}

			Ainc += Aprows;
		}

		return Q;
	}

	friend Matrix_t<T> operator*(T a, Matrix_t<T> A)
	{
		Matrix_t<T> Q (A.rows (), A.columns ());

		ssize_t drow = A.m_data->prows () - A.rows ();
		T * __restrict Araw = A.raw ();
		T * __restrict Qraw = Q.raw ();

		for (int i = A.rows (); i; --i, Araw += drow)
			for (int j = A.columns (); j; --j, ++Araw, ++Qraw)
				*Qraw = a * *Araw;

		return Q;
	}

	friend Matrix_t<T> operator+(Matrix_t<T> A, Matrix_t<T> B)
	{
		if (!MatrixView_t<T>::defined (A.get(), B.get()))
			throw ("+ dimension mismatch");
 
		Matrix_t<T> Q = prepare_target (A, B);

		T * __restrict Araw = A.raw ();
		T * __restrict Braw = B.raw ();
		T * __restrict Qraw = Q.raw ();
		ssize_t dArow = A.m_data->prows () - A.rows ();
		ssize_t dBrow = B.m_data->prows () - B.rows ();

		for (int i = A.rows (); i; --i, Araw += dArow, Braw += dBrow)
			for (int j = A.columns (); j; --j, ++Araw, ++Braw, ++Qraw)
				*Qraw = *Araw + *Braw;

		return Q;
	}

	friend Matrix_t<T> operator-(Matrix_t<T> A, Matrix_t<T> B)
	{
		if (!MatrixView_t<T>::defined (A.get(), B.get()))
			throw ("- dimension mismatch");
 
		Matrix_t<T> Q = prepare_target (A, B);
		// Matrix_t<T> Q (A.rows (), A.columns ());

		T * __restrict Araw = A.raw ();
		T * __restrict Braw = B.raw ();
		T * __restrict Qraw = Q.raw ();
		ssize_t dArow = A.m_data->prows () - A.rows ();
		ssize_t dBrow = B.m_data->prows () - B.rows ();

		for (int i = A.rows (); i; --i, Araw += dArow, Braw += dBrow)
			for (int j = A.columns (); j; --j, ++Araw, ++Braw, ++Qraw)
				*Qraw = *Araw - *Braw;

		return Q;
	}

	/*
	 * Ax = b
	 *
	 * A = GGt
	 * Solve Gy = b
	 * Solve Gtx = y
	 *
	 * b = Gy = G(Gtx) = GGtx = Ax
	 *
	 */
	bool SolveSymmetric (Matrix_t<T> &b, Matrix_t<T> &x)
	{
		int n = rows ();
		Matrix_t<T> G (n, n);

		if (!ComputeCholesky (G))
			return false;
		G.SolveLower (b, x);
		G.SolveUpper (x); // knows it really isn't transposed

		return true;
	}

	int rcount (void)
	{
		return m_data.rcount ();
	}
};

// All QR based stuff below uses Householder reflectors.

// QR decomposition, called from solveQR (), we want R to solve Ax = b
template<typename T> void Matrix_t<T>::find_R (Matrix_t<T> &b)
{
	if (!MatrixView_t<T>::defined_prod (get(), b.get())) 
		throw ("compute R, dimension mismatch");

	CoW ();
	b.CoW ();

	int rows = Matrix_t<T>::rows ();
	int columns = Matrix_t<T>::columns ();
	int prows = INVOKE->prows ();
	T beta;
	T dotVk;
	T e1 = 0; // shutup g++
	T * __restrict w = new T [rows];
	T * __restrict Vk = new T [rows];
	T * __restrict mw_data = raw ();
	T * __restrict bm_d_data = b.raw ();
	T * __restrict p;
	T * __restrict q;
	int runs = columns - 1;

	if (rows > columns)
		runs++;

	for (int k = 0; k < runs; ++k)
	{
		Vk[k] = mw_data[k * prows + k];
		dotVk = Vk[k] * Vk[k];
		e1 = dotVk;
		p = Vk + k + 1;
		q = mw_data + k + 1 + k * prows;
		for (int i = k + 1; i < rows; ++i)
		{
			*p = *q;
			dotVk += *p * *p;
			++p;
			++q;
		}

		// v = x + sign (x (1)) |x| e1
		beta = sqrt (dotVk);

		if (Vk[k] >= 0)
			Vk[k] +=  beta;
		else
			Vk[k] -=  beta;

		dotVk -= e1;
		dotVk += Vk[k] * Vk[k];
		beta = sqrt (dotVk);
		p = Vk + k;
		for (int i = k; i < rows; ++i)
			*p++ /= beta;

		// A = A + beta(-2) * vwT (w = ATv)

		for (int i = k; i < rows; ++i)
		{
			w[i] = 0;
			p = Vk + k;
			q = mw_data + i * prows + k;
			for (int j = k; j < rows; ++j)
				w[i] += *q++ * *p++;

			w[i] *= -2; // beta = -2
		}

		for (int i = k; i < rows; ++i)
			for (int j = k, index = k * prows; j < columns; ++j, index += prows)
				mw_data[index + i] += Vk[i] * w[j];

		p = Vk + k;
		q = bm_d_data + k;
		beta = 0;
		for (int i = k; i < rows; ++i)
			beta += *p++ * *q++;

		p = Vk + k;
		q = bm_d_data + k;
		beta *= 2;
		for (int i = k; i < rows; ++i)
			*q++ -= beta * *p++;
	}

	delete [] Vk;
	delete [] w;
}

// Solve for x in Rx = b, R is upper triangular (called from solveQR)
template<typename T> Matrix_t<T> Matrix_t<T>::find_x (Matrix_t<T> &b)
{
	int columns = Matrix_t<T>::columns ();
	int asym = Matrix_t<T>::rows () - columns;
	int dim = b.rows () - asym;
	Matrix_t<T> x (dim, 1);
	T sum;

	for (int i = b.rows () - 1 - asym; i >= 0; --i)
	{
		sum = 0;
		for (int j = columns - 1; j > i; --j)
			sum += (*this)(i, j) * x(j, 0);

		sum = b(i, 0) - sum;

		x(i, 0) = sum / (*this)(i, i);
	}

	return x;
}

/*
 * QR decomposition
 *
 * __JUST_V causes QR to calculate R and return the reflectors, vi, 
 * in Q instead of the qi.
 *
 */

template<typename T> void Matrix_t<T>::QR (Matrix_t<T> &Q)
{
	CoW ();
	Q = Matrix_t<T> (INVOKE->rows (), INVOKE->columns ());

	int rows = Matrix_t<T>::rows ();
	int columns = Matrix_t<T>::columns ();

	if (rows < columns)
		throw ("underdetermined system");

	int prows = INVOKE->prows ();
	T beta;
	T dotVk;
	T e1 = 0; // shutup g++
	T * __restrict w = new T [rows];
	T * __restrict Vk = new T [rows];
	T * __restrict mw_data = raw ();
	T * __restrict Q_data = Q.raw ();
	T * __restrict p;
	T * __restrict q;
	int runs = columns - 1;

	if (rows > columns)
		runs++;

	for (int k = 0; k < runs; ++k)
	{
		Vk[k] = mw_data[k * prows + k];
		dotVk = Vk[k] * Vk[k];
		e1 = dotVk;
		p = Vk + k + 1;
		q = mw_data + k + 1 + k * prows;
		for (int i = k + 1; i < rows; ++i)
		{
			*p = *q;
			dotVk += *p * *p;
			++p;
			++q;
		}

		// v = x + sign (x (1)) |x| e1
		beta = sqrt (dotVk);

		if (Vk[k] >= 0)
			Vk[k] +=  beta;
		else
			Vk[k] -=  beta;

		dotVk -= e1;
		dotVk += Vk[k] * Vk[k];
		beta = 2 / dotVk;
		p = Vk + k;

#ifdef __JUST_V
		for (int i = k; i < rows; ++i)
			Q(i, k) = Vk[i];
		Q(k, columns - 1) = beta;
#endif
		// A = A + vwT (w = ATv)

		for (int i = k; i < rows; ++i)
		{
			w[i] = 0;
			p = Vk + k;
			q = mw_data + i * prows + k;
			for (int j = k; j < rows; ++j)
				w[i] += *q++ * *p++;

			w[i] *= beta;
		}

		for (int i = k; i < rows; ++i)
			for (int j = k, index = k * prows; j < columns; ++j, index += prows)
				mw_data[index + i] -= Vk[i] * w[j];

#ifdef __JUST_V
		continue;
#endif

		// Build  Q(IT - beta*QvvT)
		// 
		// (i) construct beta Qv = w
		if (k == 0) // Q = I
			for (int r = 0; r < rows; ++r)
				w[r] = beta * Vk[r];
		else
			for (int r = 0; r < rows; ++r)
			{
				w[r] = 0;
				p = Vk + k;
				q = Q_data + r + k*prows;
				for (int c = k; c < columns; ++c)
				{
					w[r] += *q * *p;
					++p;
					q += prows;
				}

				w[r] *= beta;
			}

		// (ii) Q - wvT
		if (k == 0)
			for (int c = k; c < columns; ++c)
				for (int r = 0, index = c * prows; r < rows; ++r, ++index)
					if (r == c)
						Q_data[index] = 1 - w[r] * Vk[c];
					else
						Q_data[index] = -(w[r] * Vk[c]);
		else // k == 0
			for (int c = k; c < columns; ++c)
				for (int r = 0, index = c * prows; r < rows; ++r, ++index)
					Q_data[index] -= w[r] * Vk[c];
	}

	delete [] Vk;
	delete [] w;

	Q = Q.view (0, 0, rows, rows);
}

/*
 * Calculates the Hessenberg similarity: A = V'HV, preserves eigenvalues
 *
 */

template<typename T> void Matrix_t<T>::HessenbergSimilarity (bool similar)
{
	CoW ();

	int rows = Matrix_t<T>::rows ();
	int columns = Matrix_t<T>::columns ();
	int prows = INVOKE->prows ();
	T beta;
	T dotVk;
	T e1 = 0; // shutup g++
	T * __restrict w = new T [rows];
	T * __restrict Vk = new T [rows];
	T * __restrict mw_data = raw ();
	T * __restrict p;
	T * __restrict q;
	int runs = columns - 2;

	if (rows > columns)
		runs++;

	for (int col = 0; col < runs; ++col)
	{
		int k = col + 1;
		Vk[k] = mw_data[col * prows + k];

		dotVk = Vk[k] * Vk[k];
		e1 = dotVk;
		p = Vk + k + 1;
		q = mw_data + k + 1 + col * prows;
		for (int i = k + 1; i < rows; ++i)
		{
			*p = *q;
			dotVk += *p * *p;
			++p;
			++q;
		}

		// v = x + sign (x (1)) |x| e1
		beta = sqrt (dotVk);

		if (Vk[k] >= 0)
			Vk[k] +=  beta;
		else
			Vk[k] -=  beta;

		// fix dotVk so it is vTv and not xTx
		dotVk -= e1;
		dotVk += Vk[k] * Vk[k];
		beta = 2 / dotVk; // sqrt (dotVk);
		p = Vk + k;

		// A = A + beta(-2) * vwT (w = ATv)

		for (int i = col; i < rows; ++i)
		{
			w[i] = 0;
			p = Vk + k;
			q = mw_data + i * prows + k;
			for (int j = k; j < rows; ++j)
				w[i] += *q++ * *p++;

			w[i] *= beta;
		}

		for (int i = k; i < rows; ++i)
			for (int j = col, index = col * prows; 
				j < columns; 
				++j, index += prows)
					mw_data[index + i] -= Vk[i] * w[j];

		if (!similar)
			continue;

		/*
		 * And now from the other side _R - beta _Rvvt : continue for 
		 * Hessenberg ex eigenvalues.
		 *
		 */

		for (int r = 0; r < rows; ++r)
		{
			w[r] = 0;
			p = Vk + k;
			q = mw_data + r + k*prows;
			for (int c = k; c < columns; ++c)
			{
				w[r] += *q * *p;
				++p;
				q += prows;
			}

			w[r] *= beta;
		}

		for (int c = k; c < columns; ++c)
			for (int r = 0, index = c * prows; r < rows; ++r, ++index)
				mw_data[index] -= w[r] * Vk[c];
	}

	delete [] Vk;
	delete [] w;
}

/*
 * Given a vector (axis of reflection), v, apply the resultant Householder
 * while maintaining similarity.
 *
 */ 
template<typename T> void 
Matrix_t<T>::ApplyHouseholder (Matrix_t<T> &v, bool similar)
{
	CoW ();
	v.CoW ();

	int rows = Matrix_t<T>::rows ();
	int columns = Matrix_t<T>::columns ();
	int prows = Matrix_t<T>::prows ();
	double * __restrict w = new double [rows];
	double * __restrict Vk = v.raw (); // new double (rows);
	double * __restrict data = raw ();
	double * __restrict me = v.raw ();
	double * __restrict p;
	double * __restrict q;
	double beta;
	double dotVk;
	double e1;

	Vk[0] = me[0];

	dotVk = Vk[0] * Vk[0];
	e1 = dotVk;
	p = Vk + 1;
	q = me  + 1;
	for (int i = 1; i < rows; ++i)
	{
		*p = *q;
		dotVk += *p * *p;
		++p;
		++q;
	}

	// v = x + sign (x (1)) |x| e1
	beta = sqrt (dotVk);

	if (Vk[0] >= 0)
		Vk[0] +=  beta;
	else
		Vk[0] -=  beta;

	// fix dotVk so it is vTv and not xTx
	dotVk -= e1;
	dotVk += Vk[0] * Vk[0];
	beta = 2 / dotVk; // sqrt (dotVk);
	p = Vk;

	// A = A + beta(-2) * vwT (w = ATv)

	for (int i = 0; i < rows; ++i)
	{
		w[i] = 0;
		p = Vk;
		q = data + i * prows;
		for (int j = 0; j < rows; ++j)
			w[i] += *q++ * *p++;

		w[i] *= beta;
	}

	for (int i = 0; i < rows; ++i)
		for (int j = 0, index = 0; j < columns; ++j, index += prows)
			data[index + i] -= Vk[i] * w[j];

	if (!similar)
		return;

	/*
	 * And now from the other side _R - beta _Rvvt : continue for 
	 * Hessenberg ex eigenvalues.
	 *
	 */

	for (int r = 0; r < rows; ++r)
	{
		w[r] = 0;
		p = Vk;
		q = data + r;
		for (int c = 0; c < columns; ++c)
		{
			w[r] += *q * *p;
			++p;
			q += prows;
		}

		w[r] *= beta;
	}

	for (int c = 0; c < columns; ++c)
		for (int r = 0, index = c * prows; r < rows; ++r, ++index)
			data[index] -= w[r] * Vk[c];

	delete [] w;
}

/*
 * Given the shifts furnished in shifts(), compute and apply the resulting
 * householder and then `chase the bulge' thus returning the system
 * to a similar Hessenberg form.
 *
 * (i) p0 = (π (A - uI))e1
 * (ii) Apply p0
 * (iii) Chase the bulge accumulating the qi in Q
 *
 * Q contains the orthonormal vectors related to the operation.
 *
 */

template<typename T> void
Matrix_t<T>::ImplicitQRStep (int N, double shifts[], Matrix_t &Q)
{
	CoW ();

	/*
	 * (i) Compute p0
	 *
	 */
	int m = N + 1;
	Matrix_t<T> sigmaI (m, m, shifts[0], true);
	Matrix_t<T> sub = view (0, 0, m, m, false);
	Matrix_t<T> ax = (sub - sigmaI);

	for (int i = 1; i < N; ++i) {

		Matrix_t<T> sigmaI (m, m, shifts[i], true);
		Matrix_t<T> sub = view (0, 0, m, m, false);
		ax *= (sub - sigmaI);
	}

	Matrix_t<T> p0 (rows (), 1, 0.0);
	for (int i = 0; i < m; ++i)
		p0(i, 0) = ax(i, 0);

	/*
	 * (ii) Apply p0 (introduce the `bulge')
	 *
	 */
	ApplyHouseholder (p0, true);

	/*
	 * (iii) Return to Hessenberg form with similarity transformations
	 *
	 */
	int rows = Matrix_t<T>::rows ();
	int columns = Matrix_t<T>::columns ();
	int prows = INVOKE->prows ();
	T beta;
	T dotVk;
	T e1 = 0; // shutup g++
	T * __restrict w = new T [rows];
	T * __restrict Vk = new T [rows];
	T * __restrict mw_data = raw ();
	T * __restrict Q_data = Q.raw ();
	T * __restrict p;
	T * __restrict q;
	int runs = columns - 2;

	if (rows > columns)
		runs++;

	Vk(0) = 0;

	for (int col = 0; col < runs; ++col)
	{
		int k = col + 1;
		Vk[k] = mw_data(col * prows + k);

		dotVk = Vk[k] * Vk[k];
		e1 = dotVk;
		p = Vk + k + 1;
		q = mw_data + k + 1 + col * prows;
		for (int i = k + 1; i < rows; ++i)
		{
			*p = *q;
			dotVk += *p * *p;
			++p;
			++q;
		}

		// v = x + sign (x (1)) |x| e1
		beta = sqrt (dotVk);

		if (Vk[k] >= 0)
			Vk[k] +=  beta;
		else
			Vk[k] -=  beta;

		// fix dotVk so it is vTv and not xTx
		dotVk -= e1;
		dotVk += Vk[k] * Vk[k];
		beta = 2 / dotVk; // sqrt (dotVk);
		p = Vk + k;

		// A = A + beta(-2) * vwT (w = ATv)

		for (int i = col; i < rows; ++i)
		{
			w[i] = 0;
			p = Vk + k;
			q = mw_data + i * prows + k;
			for (int j = k; j < rows; ++j)
				w[i] += *q++ * *p++;

			w[i] *= beta;
		}

		for (int i = k; i < rows; ++i)
			for (int j = col, index = col * prows; 
				j < columns; 
				++j, index += prows)
					mw_data(index + i) -= Vk[i] * w[j];

		/*
		 * Calculate and store the qi in Q
		 *
		 * Build  Q(IT - beta*QvvT)
		 * 
		 * (i) construct beta Qv = w
		 *
		 */

		if (col == 0) // Q = I
			for (int r = 0; r < rows; ++r)
				w(r) = beta * Vk(r);
		else
			for (int r = 0; r < rows; ++r)
			{
				w(r) = 0;
				p = Vk + k;
				q = Q_data + r + k*prows;
				for (int c = k; c < columns; ++c)
				{
					w(r) += *q * *p;
					++p;
					q += prows;
				}

				w(r) *= beta;
			}

		// (ii) Q - wvT
		if (col == 0)
			for (int c = col; c < columns; ++c)
				for (int r = 0, index = c * prows; r < rows; ++r, ++index)
					if (r == c)
						Q_data(index) = 1 - w(r) * Vk(c);
					else
						Q_data(index) = -(w(r) * Vk(c));
		else // not q0
			for (int c = k; c < columns; ++c)
				for (int r = 0, index = c * prows; r < rows; ++r, ++index)
					Q_data(index) -= w(r) * Vk(c);

		/*
		 * And now from the other side _R - beta _Rvvt
		 *
		 */

		for (int r = 0; r < rows; ++r)
		{
			w(r) = 0;
			p = Vk + k;
			q = mw_data + r + k*prows;
			for (int c = k; c < columns; ++c)
			{
				w(r) += *q * *p;
				++p;
				q += prows;
			}

			w(r) *= beta;
		}

		for (int c = k; c < columns; ++c)
			for (int r = 0, index = c * prows; r < rows; ++r, ++index)
				mw_data(index) -= w(r) * Vk(c);
	}

	delete [] Vk;
	delete [] w;
}

/*
 * Compute the Cholesky decomposition.  The answer, which is lower
 * triangular, is placed in G.  G must already own memory.  It is not
 * zero'ed so elements above the diagonal will be preserved.  Doesn't matter
 * for solving SPD matrices as the code only accesses valid elements.
 *
 * To get a "clean" decomposition allocate:
 *
 * Matrix_t<T> G (n, n, 0.0, true);
 *
 */
template<typename T> bool
Matrix_t<T>::ComputeCholesky (Matrix_t<T> &G)
{
	int n = rows ();
	int step = stride ();
	double * __restrict Araw = raw ();
	double * __restrict Graw = G.raw ();

#ifdef __DEBUG
	assert (step > 0);
#endif

	for (int i = 0; i < n; ++i)
		for (int j = 0; j < (i+1); ++j) 
		{
			double s = 0;
			for (int k = 0; k < j; k++)
				s += Graw [i + k * step] * Graw[j + k * step];
			Graw[i + j * step] = (i == j) ?
				sqrt(Araw[i + i * step] - s) :
				(1.0 / Graw[j + j * step] * (Araw[i + j * step] - s));

			if (isnan (Graw[i + j * step]))
				return false;
		}

	return true;
}

template<typename T> void
Matrix_t<T>::SolveLower (Matrix_t<T> &b, Matrix_t<T> &x)
{
	double sum;
	int n = rows ();
	double *Araw = raw ();
	double *solve = x.raw ();
	double *constraint = b.raw ();
	int step = stride ();

#ifdef __DEBUG
	assert (step > 0);
#endif

	for (int i = 0; i < n; ++i)
	{
		sum = 0;

		for (int j = 0; j < i; ++j)
			sum += Araw[i + j * step] * solve[j];

		solve[i] = (constraint[i] - sum) / Araw[i + i * step];
	}
}

/*
 * Solves for the transpose.  Used in Cholesky solutions.
 *
 */

template<typename T> void
Matrix_t<T>::SolveUpper (Matrix_t<T> &b)
{
	double sum;
	int nrows = rows ();
	int n = nrows - 1;
	double *Araw = raw ();
	double *constraint = b.raw ();
	int step = stride ();

	for (int ncol = n; ncol >= 0; --ncol)
	{
		sum = 0;

		for (int nrow = n; nrow > ncol; --nrow)
			sum += Araw[nrow + ncol * step] * constraint[nrow];

		sum = constraint[ncol] - sum;
		constraint[ncol] = sum / Araw[ncol + ncol * step];
	}
}

#undef INVOKE

#endif // header inclusion

