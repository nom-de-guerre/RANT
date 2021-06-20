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

#ifndef __DJS_CNN__H__
#define __DJS_CNN__H__

#include <plane.h>
#include <layer.h>

class CNN_t
{
	int					cn_Nlayers;			// Maximum layers supported
	int					cn_N;				// Number of layers created
	int					cn_Nclasses;
	int					cn_rows;
	int					cn_columns;
	int					cn_Nsubsamples;
	int					cn_maxIterations;
	int					cn_steps;
	double				cn_haltMetric;

	layer_t				**cn_layers;

public:

	CNN_t (const int rows, 
		   const int columns,
		   const int Nlayers, 
		   const int Nclasses) :
		cn_Nlayers (Nlayers),
		cn_N (0),
		cn_Nclasses (Nclasses),
		cn_rows (rows),
		cn_columns (columns),
		cn_Nsubsamples (5000),
		cn_maxIterations (200),
		cn_steps (0),
		cn_haltMetric (0.70),
		cn_layers (new layer_t * [Nlayers])
	{
		for (int i = 0; i < cn_Nlayers; ++i)
			cn_layers[i] = NULL;
	}

	~CNN_t (void)
	{
		for (int i = 0; i < cn_N; ++i)
			if (cn_layers[i])
				delete cn_layers[i];

		delete [] cn_layers;
	}

	/*
	 * Meta parameters.
	 *
	 */

	void setSGDSamples (const int N)
	{
		cn_Nsubsamples = N;
	}

	void setMaxIterations (const int N)
	{
		cn_maxIterations = N;
	}

	void setHaltMetric (const double metric)
	{
		cn_haltMetric = metric;
	}

	/*
	 * CNN construction.
	 *
	 */

	int AddConvolutionLayer (
		const int N, 
		const int fwidth, 
		const int mwidth)
	{
		assert (cn_N < cn_Nlayers);
		const int layer = cn_N++;

		cn_layers[layer] = new layer_t (N, layer_t::CONVOLVE, fwidth, mwidth);

		return cn_layers[layer]->mapDim ();
	}

	int AddConvolutionLayerProgram (
		int Nprograms, 
		const int Nin,
		const int fwidth, 
		const int mwidth,
		const int * const stripeProgram)
	{
		assert (cn_N < cn_Nlayers);
		const int layer = cn_N++;

		cn_layers[layer] = new layer_t (Nprograms, 
			Nin, 
			fwidth, 
			mwidth, 
			stripeProgram);

		return cn_layers[layer]->mapDim ();
	}

	int AddMaxPoolSlideLayer (
		const int N, 
		const int fwidth, 
		const int mwidth)
	{
		assert (cn_N < cn_Nlayers);
		const int layer = cn_N++;

		if (layer && N > 1)
			assert (cn_layers[layer - 1]->N () == N);

		cn_layers[layer] = new layer_t (N, 
			layer_t::MAXPOOLSLIDE, 
			fwidth, 
			mwidth);

		return cn_layers[layer]->mapDim ();
	}

	int AddMaxPoolLayer (
		const int N, 
		const int fwidth, 
		const int mwidth)
	{
		assert (cn_N < cn_Nlayers);
		const int layer = cn_N++;

		if (layer && N > 1)
			assert (cn_layers[layer - 1]->N () == N);

		cn_layers[layer] = new layer_t (N, layer_t::MAXPOOL, fwidth, mwidth);

		return cn_layers[layer]->mapDim ();
	}

	bool AddFullLayer (
		int * layers,
		const int Nlayers)
	{
		assert (cn_N < cn_Nlayers);
		const int layer = cn_N++;

		layers[0] = cn_layers[layer - 1]->Nout ();

		cn_layers[layer] = new layer_t (layers, Nlayers);

		return true;
	}

	/*
	 * CNN commands.
	 *
	 */

	int Classify (plane_t *datap)
	{
		cn_layers[0]->f (datap);

		for (int i = 1; i < cn_N; ++i)
			cn_layers[i]->f (cn_layers[i - 1]);

		int answer = cn_layers [cn_N - 1]->Signal ();

		assert (answer >= 0 && answer <= cn_Nclasses);

		return answer;
	}

	bool Train (DataSet_t *p)
	{
		bool halt = false;
		cn_steps = 0;		// to support restart

		while (!halt && cn_steps < cn_maxIterations)
		{
			++cn_steps;

			TrainingStep (p);

			halt = cn_layers[cn_N - 1]->Loss () >= cn_haltMetric;

			for (int i = 0; i < cn_N; ++i)
				cn_layers[i]->UpdateWeights ();
		}

		return halt; // true --> converged
	}

	bool TrainingStep (DataSet_t *p) // this should be private...
	{
		bool success = false;

		for (int i = 0; i < cn_Nsubsamples; ++i)
		{
			int index = rand () % p->t_N;

			plane_t example (cn_rows, cn_columns, p->entry (index));
			double answer = p->Answer (index);

			cn_layers[0]->ForwardTraining (&example, answer);

			for (int j = 1; j < cn_N; ++j)
				cn_layers[j]->ForwardTraining (cn_layers[j - 1], answer);

			cn_layers[cn_N - 1]->BackwardTraining ();

			for (int j = cn_N - 1; j > 0; --j)
				cn_layers[j - 1]->BackwardTraining (cn_layers[j]);
		}

		return success;
	}

	/*
	 * Info.
	 *
	 */

	int LayerRows (const int level) const
	{
		if (level >= cn_N)
			return -1;

		return cn_layers[level]->mapDim ();
	}

	void DumpMaps (const int level)
	{
		if (level >= cn_N)
			return;

		cn_layers[level]->DumpMaps ();
	}

	int ActiveLayers (void) const
	{
		return cn_N;
	}

	int Steps (void) const
	{
		return cn_steps;
	}

	double Loss (void)
	{
		return cn_layers[cn_N - 1]->Loss ();
	}

	int *BuildDefaultProgram (int, int);
};

int *CNN_t::BuildDefaultProgram (int Nin, int N)
{
	int *program = new int [Nin * N];
	memset (program, 0, Nin * N * sizeof (int));

	assert (Nin == 6 && N == 13);

	for (int i = 0, base = 0; i < 6; ++i, base += Nin)
		for (int j = i; j < (i + 3); ++j)
			program[base + (j % Nin)] = 1;

	for (int i = 0, base = Nin * 6; i < 6; ++i, base += Nin)
		for (int j = i; j < (i + 4); ++j)
			program[base + (j % Nin)] = 1;

	for (int i = 0, base = Nin * 12; i < Nin; ++i)
		program[base + i] = 1;

	return program;
}


#endif

