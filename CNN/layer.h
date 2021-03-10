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

#ifndef __DJS_LAYER__H__
#define __DJS_LAYER__H__

#include <plane.h>

#define MAXPLANES		32

struct arg_t
{
	int			a_N;
	plane_t		*a_args[MAXPLANES];
};

struct mapAPI_t
{
	int			ma_state;
	int			ma_cache;
	int			ma_answer;

	plane_t		ma_map;

	int			ma_stripeN;
	int			*ma_program;

	// Neural network layers
	mapAPI_t () :
		ma_state (0),
		ma_cache (-1),
		ma_stripeN (-1),
		ma_program (NULL)
	{
	}

	mapAPI_t (const int dim) :
		ma_state (0),
		ma_cache (-1),
		ma_map (dim, dim),
		ma_stripeN (-1),
		ma_program (NULL)
	{
	}

	virtual ~mapAPI_t (void)
	{
		if (ma_program)
			delete [] ma_program;
	}

	void setStriped (int N)
	{
		ma_stripeN = 0;
		ma_program = new int [N];
	}

	void addProgram (int index) 
	{
		ma_program[ma_stripeN] = index;
		++ma_stripeN;
	}

	int Nstripes (void) const
	{
		return ma_stripeN;
	}

	int Answer (void)
	{
		return ma_answer;
	}

	virtual bool Forward (arg_t &) = 0;				// normal compute
	virtual bool Train (arg_t &, double) = 0;		// forward, but train
	virtual bool Backward (arg_t &) = 0;			// compute gradient
	virtual bool Update (void) = 0;					// update weights
	virtual plane_t *fetchGradient (void) = 0;

	virtual double Loss (void)
	{
		return nan (NULL);
	}

	virtual bool Halt (void)
	{
		assert (false);
	}

	int MapSize (void)
	{
		return ma_map.N ();
	}

	plane_t *getMap (void)
	{
		return &ma_map;
	}
};

typedef enum { CONVOLVE, STRIPED, MAXPOOL, FULL } Ltype_e;

#include <full.h>
#include <filter.h>
#include <MpoolStride.h>

class layer_t
{
#define LFLAGS_STRIPED				0x0001
	int				ll_flags;
	int				ll_N;			// # of components
	Ltype_e			ll_type;

	mapAPI_t		**ll_maps;
	arg_t			ll_args;

public:

	layer_t (const int N, 
			 Ltype_e type, 
			 const int subsample,
			 const int input) :
		ll_flags (0),
		ll_N (N),
		ll_type (type)
	{
		ll_maps = new mapAPI_t * [ll_N];

		for (int i = 0; i < ll_N; ++i)
		{
			switch (ll_type)
			{
			case CONVOLVE:
				ll_maps[i] = new filter_t (subsample, input);
				break;

			case MAXPOOL:
				ll_maps[i] = new Mpool_t (subsample, input);
				break;

			default:

				assert (false);
			}
		}
	}

	layer_t (const int N, 
			 const int Nin,
			 Ltype_e type, 
			 const int subsample,
			 const int input,
			 const int * const program) :
		ll_flags (LFLAGS_STRIPED),
		ll_N (N),
		ll_type (type)
	{
		assert (type == CONVOLVE);

		ll_maps = new mapAPI_t * [ll_N];

		for (int i = 0, index = 0; i < ll_N; ++i)
		{
			filter_t *p = new filter_t (subsample, input, true);
			p->setStriped (Nin);

			for (int j = 0; j < Nin; ++j, ++index)
				if (program[index])
					p->addProgram (j);

			p->Setup ();
			ll_maps[i] = p;
		}
	}

	layer_t (const int * const layers, const int Nlayers) :
		ll_flags (0),
		ll_N (1),
		ll_type (FULL)
	{
		assert (MAXPLANES >= Nlayers);

		ll_maps = new mapAPI_t *;
		ll_maps[0] = new full_t (layers, Nlayers);
	}

	~layer_t (void)
	{
		for (int i = 0; i < ll_N; ++i)
			delete ll_maps[i];

		delete [] ll_maps;
	}

	void setStriped (void)
	{
		ll_flags |= LFLAGS_STRIPED;
	}

	bool isStriped (void) const
	{
		return ((ll_flags & LFLAGS_STRIPED) == LFLAGS_STRIPED);
	}

	int N (void)
	{
		return ll_N;
	}

	bool f (plane_t *datap)
	{
		ll_args.a_N = 1;
		ll_args.a_args[0] = datap;

		for (int i = 0; i < ll_N; ++i)
			ll_maps[i]->Forward (ll_args);

		return true;
	}

	bool ForwardTraining (plane_t *datap, double answer)
	{
		ll_args.a_N = 1;
		ll_args.a_args[0] = datap;

		for (int i = 0; i < ll_N; ++i)
			ll_maps[i]->Train (ll_args, answer);

		return true;
	}

	bool ForwardTraining (layer_t *lp, double answer);

	bool BackwardTraining (void)
	{
		assert (ll_type == FULL);

		ll_maps[0]->Backward (ll_args);
	}

	bool BackwardTraining (layer_t *lp);

	int Nout (void)
	{
		return ll_N * ll_maps[0]->MapSize ();
	}

	int Answer (void)
	{
		return ll_maps[ll_N - 1]->Answer ();
	}

	int mapDim (void)
	{
		return ll_maps[0]->ma_map.rows ();
	}

	plane_t *operator[] (const int index)
	{
		return ll_maps[index]->getMap ();
	}

	void UpdateWeights (void)
	{
		for (int i = 0; i < ll_N; ++i)
			ll_maps[i]->Update ();
	}

	void DumpMaps (void)
	{
		printf ("%d\n", ll_N);
		for (int i = 0; i < ll_N; ++i)
			ll_maps[i]->ma_map.display ();
	}

	bool f (layer_t *);

	bool TestStop (void)
	{
		return ll_maps[0]->Halt ();
	}

	double Loss (void)
	{
		assert (ll_N == 1); // or what does this mean?

		return ll_maps[0]->Loss ();
	}
};

bool layer_t::f (layer_t *lp)
{
	if (isStriped ())
	{
		int *program;

		for (int i = 0; i < ll_N; ++i)
		{
			ll_args.a_N = ll_maps[i]->Nstripes ();
			int *program = ll_maps[i]->ma_program;

			for (int j = 0; j < ll_args.a_N; ++j)
				ll_args.a_args[j] = (*lp)[program[j]];

			ll_maps[i]->Forward (ll_args);
		}

	} else if (lp->N () == ll_N) {

		ll_args.a_N = 1;

		for (int i = 0; i < ll_N; ++i)
		{
			ll_args.a_args[0] = (*lp)[i];
			ll_maps[i]->Forward (ll_args);
		}

	} else {

		ll_args.a_N = lp->N ();

		for (int i = 0; i < lp->ll_N; ++i)
			ll_args.a_args[i] = (*lp)[i];

		ll_maps[0]->Forward (ll_args);
	}

	return true;
}

bool layer_t::ForwardTraining (layer_t *lp, double answer)
{
	if (isStriped ())
	{
		int *program;

		for (int i = 0; i < ll_N; ++i)
		{
			ll_args.a_N = ll_maps[i]->Nstripes ();
			int *program = ll_maps[i]->ma_program;

			for (int j = 0; j < ll_args.a_N; ++j)
				ll_args.a_args[j] = (*lp)[program[j]];

			ll_maps[i]->Train (ll_args, answer);
		}

	} else if (lp->N () == ll_N) {

		ll_args.a_N = 1;

		for (int i = 0; i < ll_N; ++i)
		{
			ll_args.a_args[0] = (*lp)[i];
			ll_maps[i]->Train (ll_args, answer);
		}

	} else {

		ll_args.a_N = lp->N ();

		for (int i = 0; i < lp->ll_N; ++i)
			ll_args.a_args[i] = (*lp)[i];

		ll_maps[0]->Train (ll_args, answer);
	}

	return true;
}

bool layer_t::BackwardTraining (layer_t *lp)
{
	if (lp->isStriped ())
	{
		assert (ll_type == MAXPOOL);
		assert (lp->ll_type == CONVOLVE);

		int *program;
		int N = lp->ll_N;

		for (int i = 0; i < N; ++i)
		{
			filter_t *p = static_cast<filter_t *>(lp->ll_maps[i]);
			ll_args.a_N = 1;
			ll_args.a_args[0] = p->fetchGradient ();
			int *program = p->ma_program;

			for (int j = 0; j < ll_args.a_N; ++j)
				ll_maps[program[j]]->Backward (ll_args);
		}

	} else if (lp->ll_type == FULL) {

		int blockSize = ll_maps[0]->MapSize ();
		plane_t *pGrad = lp->ll_maps[0]->fetchGradient ();
		plane_t dummy (1, 1, pGrad->raw ());

		ll_args.a_N = 1;
		ll_args.a_args[0] = &dummy;

		for (int i = 0; i < ll_N; ++i)
		{
			ll_maps[0]->Backward (ll_args);
			dummy.dd_datap += blockSize;
		}

	} else {

		assert (ll_N == lp->N ());

		ll_args.a_N = 1;

		for (int i = 0; i < ll_N; ++i) { // 1:1

			ll_args.a_args[0] = lp->ll_maps[i]->fetchGradient ();
			ll_maps[i]->Backward (ll_args);
		}
	}

	return true;
}

#endif // header inclusion

