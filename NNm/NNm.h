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
#include <float.h>		// FLT_EVAL_METHOD

#include <data.h>
#include <NeuralM.h>
#include <sampling.h>

#include <stratum.h>
#include <dense.h>
#include <identity.h>
#include <softmax.h>
#include <layerN.h>
#include <MSE.h>
#include <MLE.h>
#include <RPROP.h>
#include <ADAM.h>

class NNet_t
{
protected:

	int					n_steps;

	// morphology of the net
	int					n_Nin;
	int					n_Nout;
	int					n_levels;		// available slots for layers
	int					n_populated;	// slots occupied by layers

	int					*n_width;		// array of lengths of n_levels
	stratum_t			**n_strata;

	int					n_Nweights;

	IEEE_t				n_halt;			// target loss
	IEEE_t				n_error;		// current loss
	bool				n_accuracy;		// halt training at 100% correct
	int					n_maxIterations;
	int					n_keepalive;	// how often to print status

	// Stochastic Gradient Descent Implementation
	bool					n_useSGD;		// SGD turned on
	IEEE_t					n_SGDn;			// % of batch to use
	NoReplacementSamples_t	*n_SGDsamples;	// permuted samples

	// Pre-processing for the arguments
	bool				n_normalize;
	IEEE_t				*n_normParams;
	IEEE_t				*n_arg;

	bool TrainWork (const DataSet_t * const);
	bool Step(const DataSet_t * const training);
	bool Halt (DataSet_t const * const);
	
public:

#if 0
	/*
	 * levels is the number of layers, including the output.  width is an
	 * array specifying the width of each level.  e.g., { 1, 4, 1 }, is
	 * an SLP with a single input, 4 hidden and 1 output perceptron.
	 *
	 */
	NNet_t (const int levels,
			const int * const width, 
			StrategyAlloc_t rule) :
		n_steps (0),
		n_populated (levels - 1),
		n_Nin (width[0]),
		n_Nout (width[levels - 1]),
		n_levels (levels - 1), // no state for input
		n_halt (1e-5),
		n_error (nan (NULL)),
		n_accuracy (false),
		n_maxIterations (5000),
		n_keepalive (100),
		n_useSGD (false),
		n_SGDn (nan(NULL)),
		n_SGDsamples (NULL),
		n_normalize (false),
		n_normParams (NULL),
		n_arg (NULL)
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
			n_strata[i - 1] = new dense_t (i - 1, width[i], width[i - 1], rule);
			n_strata[i - 1]->_sAPI_init ();
		}
	}
#endif // deprecated constructor

	NNet_t (const int levels, const int Nin, const int Nout) :
		n_steps (0),
		n_Nin (Nin),
		n_Nout (Nout),
		n_levels (levels),
		n_populated (0),
		n_Nweights (-1),
		n_halt (1e-5),
		n_error (nan (NULL)),
		n_accuracy (false),
		n_maxIterations (5000),
		n_keepalive (100),
		n_useSGD (false),
		n_SGDn (nan(NULL)),
		n_SGDsamples (NULL),
		n_normalize (false),
		n_normParams (NULL),
		n_arg (NULL)
	{
		n_strata = new stratum_t * [n_levels];
		n_width = new int [n_levels];

		for (int i = 0; i < n_levels; ++i)
		{
			n_strata[i] = NULL;
			n_width[i] = -1;
		}
	}

	~NNet_t (void)
	{
		for (int i = 0; i < n_populated; ++i)
			delete n_strata[i];

		delete [] n_strata;

		delete [] n_width;

		if (n_normParams)
			delete [] n_normParams;

		if (n_normalize)
			delete [] n_arg;

		if (n_SGDsamples)
			delete n_SGDsamples;
	}

	int Nin (void)
	{
		return n_Nin;
	}

	int Nout (void)
	{
		return n_Nout;
	}

	void SetMaxIterations (int maxIterations)
	{
		n_maxIterations = maxIterations;
	}

	void SetHalt (IEEE_t mse)
	{
		n_halt = mse;
	}

	IEEE_t Loss (void)
	{
		return n_error;
	}

	bool TrainAndReset (DataSet_t const * const);
	bool Train (const DataSet_t * const);
	bool Train (const DataSet_t * const, int); // used only when stand-alone

	int Steps (void) const
	{
		return n_steps;
	}

	void SetAccuracy (void)
	{
		n_accuracy = true;
	}

	void TurnOffAccuracy (void)
	{
		n_accuracy = false;
	}

	void SetSGD (IEEE_t percentage)
	{
		assert (percentage > 0 && percentage <= 1.0);

		n_useSGD = true;
		n_SGDn = percentage;
	}

	void SetKeepAlive (const int modulus)
	{
		n_keepalive = modulus;
	}

	void AddDenseLayer (int N, StrategyAlloc_t rule)
	{
		int layer = n_populated++;

		assert (layer >= 0 && layer < n_levels);
		assert (n_strata[layer] == NULL);

		int Nin = (layer ? n_width[layer - 1] : n_Nin);

		n_width[layer] = N;
		n_strata[layer] = new dense_t (layer, N, Nin, rule);
		n_strata[layer]->_sAPI_init ();
	}

	void AddIdentityLayer (int N, StrategyAlloc_t rule)
	{
		int layer = n_populated++;

		assert (layer >= 0 && layer < n_levels);
		assert (n_strata[layer] == NULL);

		int Nin = (layer ? n_width[layer - 1] : n_Nin);

		n_width[layer] = N;
		n_strata[layer] = new identity_t (layer, N, Nin, rule);
		n_strata[layer]->_sAPI_init ();
	}

	// Layer normalization, not batch normalization
	void AddNormalizationLayer (StrategyAlloc_t rule)
	{
		int layer = n_populated++;

		assert (layer > 0 && layer < n_levels);
		assert (n_strata[layer] == NULL);

		int Nin = n_width[layer - 1];

		n_width[layer] = Nin;
		n_strata[layer] = new layerN_t (layer, Nin, rule);
		n_strata[layer]->_sAPI_init ();
	}

	void AddSoftmaxLayer (const int K, StrategyAlloc_t rule)
	{
		int layer = n_populated++;

		assert (layer > 0 && layer < n_levels);
		assert (n_strata[layer] == NULL);

		int Nin = n_width[layer - 1];

		n_width[layer] = K;
		n_strata[layer] = new SoftmaxMLE_t (layer, K, Nin, rule);
		n_strata[layer]->_sAPI_init ();
	}

	void AddScalerMSELayer (StrategyAlloc_t rule)
	{
		assert (n_Nout == 1);

		int layer = n_populated++;

		assert (layer > 0 && layer < n_levels);
		assert (n_strata[layer] == NULL);

		int Nin = n_width[layer - 1];

		n_width[layer] = Nin; // a 1:1 layer
		n_strata[layer] = new ScalerMSE_t (layer, Nin, rule);
		n_strata[layer]->_sAPI_init ();
	}

	IEEE_t Compute (const TrainingRow_t);
	void ComputeDerivative (const TrainingRow_t);

	// The below are public so these objects can be integrated
	void UpdateWeights (void)
	{
		for (int i = 0; i < n_populated; ++i)
			n_strata[i]->_sAPI_strategy ();
	}

	bool ExposeGradient (NeuralM_t &);

	void SetNormalizePP (DataSet_t const * const S)
	{
		assert (S->Nin () == n_Nin);

		n_normalize = true;
		n_normParams = new IEEE_t [n_Nin * 2];
		n_arg = new IEEE_t [n_Nin];

		for (int i = 0; i < n_Nin; ++i)
		{
			n_normParams[i * 2] = S->Mean (i);
			n_normParams[i * 2 + 1] = S->StdDev (i);
		}
	}

	void Reset (void)
	{
		n_error = -1;

		for (int i = 0; i < n_populated; ++i)
			n_strata[i]->_sAPI_init ();
	}

	void DisplayModel (void)
	{
		for (int i = 0; i < n_populated; ++i)
			printf ("%s (%d)\t%s",
				n_strata[i]->s_Name,
				n_strata[i]->s_Nnodes,
				(i + 1 == n_populated ? "\n" : "⟹\t"));
	}
};

#include <NNm.tcc>

#endif // header inclusion

