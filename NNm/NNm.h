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

#ifndef _NN_MATRIX__H__
#define _NN_MATRIX__H__

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include <data.h>
#include <NeuralM.h>

#define RECTIFIER(X) log (1 + exp (X))
#define SIGMOID_FN(X) (1 / (1 + exp (-X))) // derivative of rectifier

#ifdef __TANH_ACT_FN
#define ACTIVATION_FN(X) tanh(X)
#define DERIVATIVE_FN(Y) (1 - Y*Y)
#else
#define ACTIVATION_FN(X) SIGMOID_FN(X)
#define DERIVATIVE_FN(Y) (Y * (1 - Y))
#endif

// RPROP+ update parameters
#define DELTA0				1e-2
#define DELTA_MIN			1e-8
#define DELTA_MAX			50

#define ETA_PLUS			1.2
#define ETA_MINUS			0.5

#ifndef SIGN
#define SIGN(X) (signbit (X) != 0 ? -1.0 : 1.0)
#endif

/*
 * Implements a layer when training a neural network.
 *
 */
struct stratum_t
{
	int						s_Nperceptrons;
	int						s_Nin; 				// # weights, includes bias

	// Matrices - per weight, s_Nperceptrons x s_Nin
	NeuralM_t					s_W;
	NeuralM_t					s_Ei;
	NeuralM_t					s_dL;
	NeuralM_t					s_deltaW;

	// Vectors - per perceptron (node)
	NeuralM_t					s_delta;
	NeuralM_t					s_dot;
	NeuralM_t					s_response;

	stratum_t (const int N, const int Nin) :
		s_Nperceptrons (N),
		s_Nin (Nin + 1),				// account for bias
		s_W (s_Nperceptrons, s_Nin),
		s_Ei (s_Nperceptrons, s_Nin),
		s_dL (s_Nperceptrons, s_Nin),
		s_deltaW (s_Nperceptrons, s_Nin),
		s_delta (s_Nperceptrons, 1),
		s_dot (s_Nperceptrons, 1),
		s_response (s_Nperceptrons, 1)
	{
		s_Ei.zero ();
		s_dL.zero ();
		s_deltaW.zero ();
	}

	~stratum_t (void)
	{
	}

	void init (int Nout)
	{
		// Glorot, W ~ [-r, r]
		double r = sqrt (6.0 / (Nout + s_Nin));
		double *p = s_W.raw();
		double *deltaW = s_deltaW.raw ();
		double sample;

		for (int i = s_W.rows () - 1; i >= 0; --i)
			for (int j = s_W.columns () - 1; j >= 0; --j)
			{
				sample = (double) rand () / RAND_MAX;
				sample *= r;
				if (rand () % 2)
					sample = -sample;
				*p++ = sample;

				*deltaW++ = DELTA0;
			}
	}

	int N (void)
	{
		return s_Nperceptrons;
	}

	void bprop (stratum_t &, double *);
	void RPROP (void);
	void RPROP (int);
	double *f (double *, bool = true);
	double *f (double *, double *);
};

template<typename T> class NNet_t
{
protected:

	int					n_steps;

	// morphology of the net
	int					n_Nin;
	int					n_Nout;
	int					n_levels;
	int					*n_width;		// array of lengths of n_nn
	stratum_t			**n_strata;

	int					n_Nweights;

	double				n_halt;			// solution accuracy (sumsq, not derivs)
	double				n_error;

	bool TrainWork (const DataSet_t * const, int);
	bool Step(const DataSet_t * const training);
	bool Halt (DataSet_t const * const);
	
public:

	/*
	 * levels is the number of layers, including the output.  width is an
	 * array specifying the width of each level.  e.g., { 1, 4, 1 }, is
	 * an SLP with a single input, 4 hidden and 1 output perceptron.
	 *
	 */
	NNet_t (const int * const width, const int levels) :
		n_steps (0),
		n_Nin (width[0]),
		n_Nout (width[levels - 1]),
		n_levels (levels - 1), // no state for input
		n_halt (1e-5),
		n_error (nan (NULL))
	{
		n_Nweights = 0;

		// width = # inputs, width 1, ..., # outputs

		for (int i = levels - 1; i > 0; --i)
			n_Nweights += width[i] * (width[i - 1] + 1); // + 1 for bias

		n_strata = new stratum_t * [n_levels];
		n_width = new int [n_levels];

		printf ("In\tOut\tTrainable\n");
		printf ("%d\t%d\t%d\n", n_Nin, n_Nout, n_Nweights);

		// start at 1, ignore inputs
		for (int i = 1; i <= n_levels; ++i)
		{
			n_width[i - 1] = width[i];
			n_strata[i - 1] = new stratum_t (width[i], width[i - 1]);
			n_strata[i - 1]->init (i < n_levels ? width[i + 1] : width[i]);
		}
	}

	~NNet_t (void)
	{
		for (int i = 0; i < n_levels; ++i)
			delete n_strata[i];

		delete [] n_strata;
	}

	void SetHalt (double mse)
	{
		n_halt = mse;
	}

	bool Train (const DataSet_t * const, int); // used only when stand-alone

	int Steps (void) const
	{
		return n_steps;
	}

	/*
	 * The next 4 call the specialization.  They are public as
	 * CNN components need to access them to integrate training.
	 *
	 */
	void Start (void);								// calls Cycle
	double Compute (double *);						// calls f ()
	inline double Loss (DataSet_t const *);			// calls Error
	double ComputeDerivative (const TrainingRow_t);	// calls bprop

	// The below are public so these objects can be integrated
	void UpdateWeights (void)
	{
		for (int i = 0; i < n_levels; ++i)
			n_strata[i]->RPROP ();
	}

	bool ExposeGradient (NeuralM_t &);
};

#include <NNm.tcc>

#endif // header inclusion

