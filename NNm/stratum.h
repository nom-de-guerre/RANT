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

#ifndef _NN_STRATUM__H__
#define _NN_STRATUM__H__

#include <string.h>
#include <stdlib.h>

#define MAXLAYERNAME 32			// Maximum length of the name of a layer type

#include <NeuralM.h>
#include <strategy.h>
#include <shape.h>

struct stratum_t : shape_t
{
	char					*s_Name;
	const int				s_ID;

	int						s_Nnodes;
	int						s_Nin;

	NeuralM_t				s_delta;
	NeuralM_t				s_response;

	strategy_t				*s_strat;

	bool					s_skip;			// abstract layer
	bool					s_frozen;		// update during training

	stratum_t (const char *namep, const int ID, const shape_t &X) :
		shape_t (X),
		s_Name (namep ? strdup (namep) : NULL),
		s_ID (ID),
		s_Nnodes (X.sh_N),
		s_Nin (X.len ()),
		s_delta (X.sh_N * X.sh_rows, X.sh_columns),
		s_response (X.sh_N * X.sh_rows, X.sh_columns),
		s_strat (NULL),
		s_skip (false),
		s_frozen (true)
	{
		s_delta.zero ();
		s_response.zero ();
	}

	stratum_t (const char *namep, const int ID, const int N, const int Nin) :
		shape_t (N),
		s_Name (namep ? strdup (namep) : NULL),
		s_ID (ID),
		s_Nnodes (N),
		s_Nin (Nin),
		s_delta (N, 1),
		s_response (N, 1),
		s_strat (NULL),
		s_skip (false),
		s_frozen (true)
	{
		s_delta.zero ();
		s_response.zero ();
	}

	// Used for loading a model from a file
	stratum_t (const char *namep) :
		shape_t (-1),
		s_Name (namep ? strdup (namep) : NULL),
		s_ID (-1),
		s_Nnodes (-1),
		s_Nin (-1),
		s_delta (),
		s_response (),
		s_strat (NULL),
		s_skip (false),
		s_frozen (true)
	{
	}

	virtual ~stratum_t (void)
	{
		if (s_Name)
			free (s_Name);
	}

	int N (void) const
	{
		return s_Nnodes;
	}

	inline bool skip (void) const
	{
		return s_skip;
	}

	void Thaw (void)
	{
		s_frozen = false;
	}

	void Freeze (void)
	{
		s_frozen = true;
	}

	void InitLearnable (const int N, IEEE_t *learnable)
	{
		IEEE_t r = sqrt (6.0 / (s_response.rows () + s_Nin));
		IEEE_t sample;
		IEEE_t *p = learnable;

		for (int i = 0; i < N; ++i)
		{
			sample = (IEEE_t) rand () / RAND_MAX;
			sample *= r;

			if (rand () % 2)
				sample = -sample;

			*p++ = sample;
		}
	}

	virtual void _sAPI_init (void) = 0;
	virtual IEEE_t * _sAPI_f (IEEE_t * const, bool = true) = 0;

	// Place dL/dz in the caller s_delta
	virtual void _sAPI_gradient (stratum_t &) = 0;

	// Update learnable parameters.  s_delta contains dL/dz: NOT delta
	virtual void _sAPI_bprop (IEEE_t *, bool = true) = 0;

	virtual void _sAPI_strategy (void)
	{
		if (s_strat)
			s_strat->_tAPI_strategy ();
	}

	virtual NeuralM_t * _sAPI_gradientM (void)
	{
		return &s_delta;
	}

	virtual int _sAPI_Store (FILE *fp)
	{
		assert (false);

		return 0;
	}

	/*
	 * This should only be called on loss layers, such as MSE,
	 * and they must override it.
	 *
	 * It assumes that prior to invocation _sAPI_f has been called.
	 *
	 */
	virtual IEEE_t _sAPI_Loss (IEEE_t const * const)
	{
		assert (false);
	}

	// returns the number of learnable parameters in the layer
	virtual int _sAPI_Trainable (void)
	{
		return 0;
	}

	virtual void _sAPI_DumpMaps (FILE *fp = stdout)
	{
		shape_t::Display ("@Shape");
		s_response.displayExp ("@Map", fp);

		return;
	}

	IEEE_t * z (void)
	{
		return s_response.raw ();
	}

	shape_t &GetShape (void)
	{
		return (shape_t &) *this;
	}
};

#endif // header inclusion

