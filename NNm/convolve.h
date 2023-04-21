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

#ifndef _NN_CONVOLUTION__H__
#define _NN_CONVOLUTION__H__

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include <NNm.h>

/*
 * The API to the convolution class.
 *
 */

struct discrete_t
{
	discrete_t (void)
	{
	}

	~discrete_t (void)
	{
	}

	virtual void f (IEEE_t *, IEEE_t *) = 0; 
	virtual void UpdateWeights (void) = 0;
	virtual void BPROP (IEEE_t *, IEEE_t *) = 0;
	virtual void Propagate (IEEE_t *, IEEE_t *) = 0;

	virtual int Persist (FILE *, const int)
	{
		assert (false);
	}
};

/*
 * Implements a layer when training a neural network.
 *
 */

template <typename T>
struct convolve_t : public stratum_t
{
	T						**cn_maps;

	int						cn_fwidth;

	int						cn_imapSize;
	bool					cn_oneToOne;

	typedef enum { INVALID, SLIDE, JUMP } e_mode;
	e_mode					cn_mode;

	convolve_t (const int ID,
				const char *type,
				const int N,
				const int fwidth,
				const shape_t Xin,
				StrategyAlloc_t rule,
				const e_mode mode = SLIDE) : 
		stratum_t (type,
				ID,
				shape_t (N, 
						(mode == SLIDE ?
						 Xin.sh_rows - fwidth + 1 :
						 Xin.sh_rows / fwidth),
						(mode == SLIDE ?
						 Xin.sh_columns - fwidth + 1 :
						 Xin.sh_columns / fwidth))),
		cn_fwidth (fwidth),
		cn_imapSize (Xin.mapSize ()),
		cn_oneToOne (Xin.sh_N > 1)
	{
		cn_maps = new T * [sh_N];

		for (int i = 0; i < sh_N; ++i)
			cn_maps[i] = new T (cn_fwidth, Xin.sh_rows, rule);
	}

	convolve_t (FILE *fp, const char *type) : stratum_t (type)
	{
		char buffer[32];

		int rc = fscanf (fp, "%s %d\t%d\t%d\t%d\n",
			buffer,
			&cn_fwidth,
			&cn_imapSize,
			(int *) &cn_oneToOne,
			&cn_mode);

		assert (rc == 5);
		assert (strcmp (buffer, "@Spensa") == 0);

		rc = fscanf (fp, "%s %d\t%d\t%d\n",
			buffer,
			&sh_N,
			&sh_rows,
			&sh_columns);

		assert (rc == 4);
		assert (strcmp (buffer, "@Shape") == 0);

		sh_length = sh_rows * sh_columns;

		s_Nnodes = sh_N;
		s_response.resize (sh_N * sh_rows * sh_columns, 1);

		cn_maps = new T * [sh_N];

		for (int i = 0; i < sh_N; ++i)
			cn_maps[i] = new T (fp, i, cn_fwidth, sh_rows);
	}

	virtual ~convolve_t (void)
	{
		for (int i = 0; i < sh_N; ++i)
			delete cn_maps[i];

		delete cn_maps;
	}

	int N (void) const
	{
		return sh_length;
	}

	void _sAPI_init (void)
	{
		// done in the discrete_t constructor
	}

	virtual IEEE_t * _sAPI_f (IEEE_t * const, bool = true);
	virtual void _sAPI_gradient (stratum_t &);
	virtual void _sAPI_bprop (IEEE_t *, bool = true);

	virtual void _sAPI_strategy (void)
	{
		for (int i = 0; i < sh_N; ++i)
			cn_maps[i]->UpdateWeights ();
	}

	virtual int _sAPI_Trainable (void)
	{
		return sh_N * cn_fwidth * cn_fwidth + sh_N;
	}

	virtual int _sAPI_Store (FILE *fp)
	{
		fprintf (fp, "@%s\n", s_Name);

		fprintf (fp, "@Spensa %d\t%d\t%d\t%d\n",
			cn_fwidth,
			cn_imapSize,
			cn_oneToOne,
			cn_mode);

		fprintf (fp, "@Shape %d\t%d\t%d\n",
			sh_N,
			sh_rows,
			sh_columns);

		for (int i = 0; i < sh_N; ++i)
			cn_maps[i]->Persist (fp, i);

		return 0;
	}

	virtual void StrategyMono (const int index)
	{
		return; // over-ride when debugging or instrumenting
	}

	void DumpMaps (void)
	{
		s_response.display ();
	}
};

template <typename T> void
convolve_t<T>::_sAPI_gradient (stratum_t &Z)
{
	IEEE_t *targetp = Z.s_delta.raw ();
	IEEE_t *gradp = s_delta.raw ();

	Z.s_delta.zero ();

	for (int i = 0; i < sh_N; ++i)
	{
		cn_maps[i]->Propagate (gradp, targetp);

		gradp += mapSize ();
		if (cn_oneToOne)
			targetp += cn_imapSize;
	}
}

template <typename T> void 
convolve_t<T>::_sAPI_bprop (IEEE_t *xi, bool activation)
{
	/*
	 *
	 * We are called with,
	 *
	 *    ∂L
	 *  ∑ -- ,
	 *    ∂z
	 *
	 * in s_delta.
	 *
	 */

	IEEE_t *inputp = xi;
	IEEE_t *gradp = s_delta.raw ();
	int block = mapSize ();

	for (int i = 0; i < sh_N; ++i)
	{
		cn_maps[i]->BPROP (gradp, inputp);

		gradp += block;
		if (cn_oneToOne)
			inputp += cn_imapSize;
	}
}

template <typename T> IEEE_t *
convolve_t<T>::_sAPI_f (IEEE_t * const xi, bool activate)
{
	IEEE_t *inputp = xi;
	IEEE_t *outputp = s_response.raw ();
	int block = sh_rows * sh_columns;

	for (int i = 0; i < sh_N; ++i)
	{
		cn_maps[i]->f (inputp, outputp);

		outputp += block;
		if (cn_oneToOne)
			inputp += cn_imapSize;
	}

	return s_response.sm_data;
}

#include <filter.h>
#include <Mpool.h>

#endif // header inclusion

