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

#ifndef _NN_NNET_ANT_BASE__H__
#define _NN_NNET_ANT_BASE__H__

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>		// FLT_EVAL_METHOD

#include <data.h>
#include <NeuralM.h>
#include <sampling.h>

// Layers
#include <stratum.h>
#include <dense.h>
#include <identity.h>
#include <softmax.h>
#include <layerN.h>
#include <dropout.h>
#include <convolve.h>

struct verify_t;

#include <MSE.h>
#include <MLE.h>
#include <multiLabel.h>

// Strategies
#include <RPROP.h>
#include <ADAM.h>

class NNet_t
{
protected:

	int					n_steps;

	// morphology of the net
	int					n_Nin;
	int					n_Nout;
	int					n_levels;			// available slots for layers
	int					n_populated;		// slots occupied by layers

	int					*n_width;			// array of lengths of n_levels
	stratum_t			**n_strata;

	int					n_Nweights;

	IEEE_t				n_halt;				// target loss

	IEEE_t				n_error;			// current loss

	int					n_accuracy;			// N correct in epoch
	bool				n_HaltOnAccuracy;	// halt training at 100% correct

	int					n_maxIterations;
	int					n_keepalive;		// how often to print status

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
	
	void LoadModel (const char *);

public:

	/*
	 * Use this constructor to load a model from a file.
	 *
	 * It will not be further trainable.
	 *
	 */
	NNet_t (const char *filename) :
		n_steps (-1),
		n_Nin (-1),
		n_Nout (-1),
		n_levels (-1),
		n_populated (0),
		n_width (NULL),
		n_strata (NULL),
		n_Nweights (-1),
		n_halt (1e-5),
		n_error (nan ("")),
		n_HaltOnAccuracy (false),
		n_maxIterations (5000),
		n_keepalive (-1),
		n_useSGD (false),
		n_SGDn (nan("")),
		n_SGDsamples (NULL),
		n_normalize (false),
		n_normParams (NULL),
		n_arg (NULL)
	{
		LoadModel (filename);
	}

	NNet_t (const int levels, const int Nin, const int Nout) :
		n_steps (0),
		n_Nin (Nin),
		n_Nout (Nout),
		n_levels (levels),
		n_populated (0),
		n_Nweights (-1),
		n_halt (1e-5),
		n_error (nan ("")),
		n_HaltOnAccuracy (false),
		n_maxIterations (5000),
		n_keepalive (100),
		n_useSGD (false),
		n_SGDn (nan("")),
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

		if (n_width)
			delete [] n_width;

		if (n_normParams)
			delete [] n_normParams;

		if (n_normalize)
			delete [] n_arg;

		if (n_SGDsamples)
			delete n_SGDsamples;
	}

	int Nin (void) const
	{
		return n_Nin;
	}

	int Nout (void) const
	{
		return n_Nout;
	}

	int Nparameters (void) const
	{
		int N = 0;

		for (int i = 0; i < n_populated; ++i)
			N += n_strata[i]->_sAPI_Trainable ();

		return N;
	}

	void SetMaxIterations (int maxIterations)
	{
		n_maxIterations = maxIterations;
	}

	void SetHalt (IEEE_t mse)
	{
		n_halt = mse;
	}

	IEEE_t Loss (void) const
	{
		return n_error;
	}

	IEEE_t Accuracy (void) const
	{
		return nan ("");
	}

	bool TrainAndReset (DataSet_t const * const);
	bool Train (const DataSet_t * const);
	bool Train (const DataSet_t * const, int); // used only when stand-alone

	void Thaw (void)
	{
		for (int i = 0; i < n_populated; ++i)
			n_strata[i]->Thaw ();
	}

	void Freeze (void)
	{
		for (int i = 0; i < n_populated; ++i)
			n_strata[i]->Freeze ();
	}

	int Steps (void) const
	{
		return n_steps;
	}

	void SetAccuracy (void)
	{
		n_HaltOnAccuracy = true;
	}

	void TurnOffAccuracy (void)
	{
		n_HaltOnAccuracy = false;
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

		int Nin = (layer ? n_strata[layer - 1]->len () : n_Nin);
		n_width[layer] = N;
		n_strata[layer] = new dense_t (layer, N, Nin, rule);
		n_strata[layer]->_sAPI_init ();
	}

	/*
	 * Layers that comprise an ANN.
	 *
	 */

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

	void AddDropoutLayer (IEEE_t p_retain)
	{
		int layer = n_populated++;

		assert (layer > 0 && layer < n_levels);
		assert (n_strata[layer] == NULL);

		int Nin = n_width[layer - 1];

		n_width[layer] = Nin;
		n_strata[layer] = new dropout_t (Nin, p_retain);
		n_strata[layer]->_sAPI_init ();
	}

	void AddFilterLayer (int N, int fwidth, StrategyAlloc_t rule)
	{
		if (N < 1)
			return;

		int layer = n_populated++;

		assert (layer >= 0 && layer < n_levels);
		assert (n_strata[layer] == NULL);

		shape_t Xin;

		if (layer == 0) {

			double rows = sqrt (n_Nin);

			assert (floor (rows) == ceil (rows));

			Xin = shape_t (1, sqrt (n_Nin), sqrt (n_Nin));

		} else {

			Xin = n_strata[layer - 1]->GetShape ();
			if (Xin.isFlat ())
			{
				IEEE_t dim = sqrt (Xin.sh_rows);

				assert (floor (dim) == ceil (dim));

				Xin.sh_columns = (int) sqrt (Xin.sh_rows);
				Xin.sh_rows = Xin.sh_columns;
			}
		}

		auto p = new convolve_t<filter_t> (
			layer, 
			"filter", 
			N, 
			fwidth, 
			Xin, 
			rule);
		n_strata[layer] = p;
		n_width[layer] = p->N ();
		n_strata[layer]->_sAPI_init ();
	}

	void AddMaxPoolLayer (int N, int fwidth)
	{
		if (N < 1)
			return;

		int layer = n_populated++;

		assert (layer >= 0 && layer < n_levels);
		assert (n_strata[layer] == NULL);

		shape_t Xin;

		if (layer == 0) {

			Xin = shape_t (1, sqrt (n_Nin), sqrt (n_Nin));

		} else {

			Xin = n_strata[layer - 1]->GetShape ();
			if (Xin.isFlat ())
			{
				IEEE_t dim = sqrt (Xin.sh_rows);

				assert (floor (dim) == ceil (dim));

				Xin.sh_columns = (int) sqrt (Xin.sh_rows);
				Xin.sh_rows = Xin.sh_columns;
			}
		}

		auto *p = new convolve_t<Mpool_t> (
			layer,
			"Maxpool",
			N,
			fwidth,
			Xin,
			NULL,
			convolve_t<Mpool_t>::e_mode::JUMP);

		n_strata[layer] = p;
		n_width[layer] = p->N ();
		n_strata[layer]->_sAPI_init ();
	}

	void AddVerificationLayer (void);

	/*
	 * Loss layers.
	 *
	 */

	// Multiclass classification - cross-entropy loss
	void AddSoftmaxLayer (StrategyAlloc_t rule)
	{
		int layer = n_populated++;

		assert (layer >= 0 && layer < n_levels);
		assert (n_strata[layer] == NULL);

		int Nin = (layer == 0 ? n_Nin :  n_width[layer - 1]);

		n_width[layer] = n_Nout;
		n_strata[layer] = new SoftmaxMLE_t (layer, n_Nout, Nin, rule);
		n_strata[layer]->_sAPI_init ();
	}

	// Regression - mean-squared error loss
	void AddScalerMSELayer (StrategyAlloc_t rule)
	{
		assert (n_Nout == 1);

		int layer = n_populated++;

		assert (layer >= 0 && layer < n_levels);
		assert (n_strata[layer] == NULL);

		int Nin = n_width[layer - 1];

		n_width[layer] = 1;
		n_strata[layer] = new ScalerMSE_t (layer, Nin, rule);
		n_strata[layer]->_sAPI_init ();
	}

	// Multilabel classification - per-label sigmoid and MSE
	void AddMultiCLayer (StrategyAlloc_t rule)
	{
		int layer = n_populated++;

		assert (layer >= 0 && layer < n_levels);
		assert (n_strata[layer] == NULL);

		int Nin = (layer ? n_width[layer - 1] : n_Nin);

		n_width[layer] = n_Nout;
		n_strata[layer] = new multiL_t (layer, n_Nout, Nin, rule);
		n_strata[layer]->_sAPI_init ();
	}

	IEEE_t *ComputeWork (const TrainingRow_t);

	/*
	 * Used for multilabel classification.
	 *
	 */
	IEEE_t * const ComputeVec (const TrainingRow_t x)
	{
		return ComputeWork (x);
	}

	/*
	 * Scaler.  Used for regression and classification.
	 *
	 */
	IEEE_t Compute (const TrainingRow_t x)
	{
		IEEE_t *y;

		y = ComputeWork (x);

		return *y;
	}

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

	IEEE_t IndividualLoss (IEEE_t answer)
	{
		return n_strata[n_populated - 1]->_sAPI_Loss (&answer);
	}

	verify_t *DifferencingLayer (const int level);

	stratum_t *Bottom (void)
	{
		return n_strata[n_populated - 1];
	}

	void Reset (void)
	{
		n_error = -1;

		for (int i = 0; i < n_populated; ++i)
			n_strata[i]->_sAPI_init ();
	}

	/*
	 * Save a trained model to a file.
	 *
	 */
	bool SaveModel (const char *file, bool overwrite=true)
	{
		FILE *fp;

		if (overwrite)
			 fp = fopen (file, "w+");
		else
			 fp = fopen (file, "w+x");

		if (fp == NULL)
			return errno;

		fprintf (fp, "@Version\t1.0\n");
		fprintf (fp, "@Topology\t%d\t%d\t%d\n",
			n_Nin,
			n_populated,
			n_Nout);

		if (n_normalize)
		{
			fprintf (fp, "@Preprocess\n");

			for (int i = 0; i < n_Nin; ++i)
				fprintf (fp, "\t%e\t%e\n",
					n_normParams[2 * i],
					n_normParams[2 * i + 1]);
		}

		fprintf (fp, "@Layers\n");

		for (int i = 0; i < n_populated; ++i)
			n_strata[i]->_sAPI_Store (fp);

		fclose (fp);

		return true;
	}

	/*
	 * Read the pre-processing parameters for a trained ANN from a saved model.
	 *
	 * Called from LoadModel.
	 *
	 */
	int LoadPreprocessing (FILE *fp)
	{
		n_normalize = true;
		n_normParams = new IEEE_t [n_Nin * 2];
		n_arg = new IEEE_t [n_Nin];

		for (int i = 0; i < n_Nin; ++i)
		{
			int rc = fscanf (fp, "%le\t%le\n",
				n_normParams + (2 * i),
				n_normParams + (2 * i) + 1);
			if (rc != 2)
				throw ("Bad PP Layer");
		}

		return 0;
	}

	/*
	 * Print the info for each layer in the ANN.
	 *
	 */
	void DisplayModel (void) const
	{
		printf ("Input (%d) ⟹ ", n_Nin);

		for (int i = 0; i < n_populated; ++i)
			printf ("%s (%d)\t%s",
				n_strata[i]->s_Name,
				n_strata[i]->s_Nnodes,
				(i + 1 == n_populated ? "\n" : "⟹ "));
	}

	/*
	 * Print the output shape of each layer.
	 *
	 */
	void DisplayShape (void) const
	{
		for (int i = 0; i < n_populated; ++i)
			n_strata[i]->GetShape ().Display ();
	}

	/*
	 * Print the map for a CNN or a Maxpool layer.
	 *
	 */
	void DumpMaps (const int level) const
	{
		if (level < 0 || level >= n_populated)
			return;

		auto fp = dynamic_cast<convolve_t <filter_t> *> (n_strata[level]);
		if (fp != NULL)
		{
			fp->DumpMaps ();

			return;
		}

		auto mp = dynamic_cast<convolve_t <Mpool_t> *> (n_strata[level]);
		if (mp == NULL)
			return;

		mp->DumpMaps ();
	}

	friend verify_t;
};

/*
 * Code to verify layers with differencing equations.
 *
 * BEWARE OF DISCONTINUITIES
 *
 */

#include <verify.h>

void NNet_t::AddVerificationLayer (void)
{
	int layer = n_populated++;

	assert (layer >= 0 && layer < n_levels);
	assert (n_strata[layer] == NULL);

	n_width[layer] = (layer == 0 ? n_Nin : n_width[layer - 1]);
	n_strata[layer] = new verify_t (layer, n_width[layer]);
}

verify_t *NNet_t::DifferencingLayer (const int level)
{
	if (level < 0 || level >= n_populated)
		return NULL;

	verify_t *fp = dynamic_cast<verify_t *> (n_strata[level]);
	return fp;
}

#include <NNm.tcc>

#endif // header inclusion

