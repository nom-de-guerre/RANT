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

IEEE_t
NNet_t::Compute (IEEE_t *x)
{
	IEEE_t *ripple;

	if (n_normalize) {

		for (int i = 0; i < n_Nin; ++i)
		{
			n_arg[i] = x[i] - n_normParams[2 * i];
			n_arg[i] /= n_normParams[2 * i + 1];
		}

	} else
		n_arg = x;

	ripple = n_arg;

	for (int layer = 0; layer < n_populated; ++layer)
		ripple = n_strata[layer]->_sAPI_f (ripple);

	return ripple[0];
}

void
NNet_t::ComputeDerivative (const TrainingRow_t x)
{
	IEEE_t *Xi;

	/*
	 * Initiate the recurrence by triggering the loss function.
	 *
	 */
	Compute (x);

	n_error += n_strata[n_populated - 1]->_sAPI_Loss (x + n_Nin);
	if (n_populated > 1)
		n_strata[n_populated - 1]->_sAPI_bprop (n_strata[n_populated - 2]->z ());

	for (int level = n_populated - 2; level >= 0; --level)
	{
		Xi = (level > 0 ? n_strata[level - 1]->z () : n_arg);

		n_strata[level + 1]->_sAPI_gradient (*n_strata[level]);

		n_strata[level]->_sAPI_bprop (Xi);
	}
}

/*
 * ------------- The generic ANN Code ---------------
 *
 */

bool
NNet_t::TrainAndReset (DataSet_t const * const training)
{
	Reset ();

	return Train (training);
}

bool
NNet_t::Train (const DataSet_t * const training)
{
	bool rc;

	rc = TrainWork (training);

	return rc;
}

bool 
NNet_t::Train (const DataSet_t * const training, int maxIterations)
{
	bool rc;

	n_maxIterations = maxIterations;
	rc = TrainWork (training);

	return rc;
}

bool 
NNet_t::TrainWork (const DataSet_t * const training)
{
	bool solved = false;

	if (n_useSGD)
		n_SGDsamples = new NoReplacementSamples_t (training->N ());

	Thaw ();

	for (n_steps = 0;
		(n_steps < n_maxIterations) && !solved;
		++n_steps)
	{
		n_error = 0.0;

		try {

			solved = Step (training);

		} catch (const char *error) {

			printf ("%s\tstill %d steps to try.\n",
				error,
				n_maxIterations - n_steps);
		}

		if (n_steps && n_keepalive && (n_steps % n_keepalive) == 0)
			printf ("(%d) Training Loss: %e\n", n_steps, Loss ());
	}

	Freeze ();

	if (!solved)
		return false;

	return true;
}

bool 
NNet_t::Step (const DataSet_t * const training)
{
	int batch = training->N ();
	int sample;
	bool done;

	if (n_useSGD)
		batch = n_SGDn * batch;

	for (int i = 0; i < batch; ++i)
	{
		(n_useSGD ? sample = n_SGDsamples->SampleAuto () : sample = i);
		assert (sample >= 0 && sample < training->N ());
		ComputeDerivative (training->entry (sample));
	}

	n_error /= batch;

	done = n_error <= n_halt;
	if (!done)
		UpdateWeights ();

	return done;
}

