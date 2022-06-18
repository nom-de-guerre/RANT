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

/*
 * Interface to loss function.
 *
 */
template<typename T> void 
NNet_t<T>::Start (void)
{
	// Starting a new batch
	return static_cast<T *> (this)->_API_Cycle ();
}

template<typename T> bool
NNet_t<T>::Halt (DataSet_t const * const tp)
{
	// Finished a batch, can we stop?
	return static_cast<T *> (this)->_API_Test (tp);
}

template<typename T> IEEE_t
NNet_t<T>::Loss (void)
{
	// The current value of the loss function
	return static_cast<T *> (this)->_API_Error ();
}

template<typename T> IEEE_t
NNet_t<T>::ComputeDerivative (const TrainingRow_t x)
{
	/*
	 * Initiate the recurrence by triggering the loss function.
	 *
	 */
	IEEE_t *Xi;
	NeuralM_t *gradp;

	IEEE_t error;

	for (int level = n_populated - 1; level >= 0; --level)
	{
		Xi = (level > 0 ? n_strata[level - 1]->z () : x);
		gradp = n_strata[level]->_sAPI_gradientM ();

		if (level == n_populated - 1)
			error = static_cast<T *>(this)->_API_bprop (x, gradp->raw ());
		else
			n_strata[level + 1]->_sAPI_gradient (*n_strata[level]);

		n_strata[level]->_sAPI_bprop (Xi);
	}

	return error;
}

template<typename T> IEEE_t
NNet_t<T>::Compute (IEEE_t *x)
{
	IEEE_t *ripple;

	if (n_normalize) {

		for (int i = 0; i < n_Nin; ++i)
		{
			n_arg[i] = x[i] - n_normParams[2 * i];
			n_arg[i] /= n_normParams[2 * i + 1];
		}

		ripple = n_arg;

	} else
		ripple = x;

	for (int layer = 0; layer < n_populated - 1; ++layer)
		ripple = n_strata[layer]->_sAPI_f (ripple);

	/*
	 * Compute the final result with the specialization
	 *
	 */
	return static_cast<T *>(this)->_API_f (ripple);
}

/*
 * ------------- The generic ANN Code ---------------
 *
 */

template<typename T> bool
NNet_t<T>::Train (const DataSet_t * const training)
{
	bool rc;

	rc = TrainWork (training);

	return rc;
}

template<typename T> bool 
NNet_t<T>::Train (const DataSet_t * const training, int maxIterations)
{
	bool rc;

	n_maxIterations = maxIterations;
	rc = TrainWork (training);

	return rc;
}

template<typename T> bool 
NNet_t<T>::TrainWork (const DataSet_t * const training)
{
	bool solved = false;

	if (n_useSGD)
		n_SGDsamples = new NoReplacementSamples_t (training->N ());

	for (n_steps = 0;
		(n_steps < n_maxIterations) && !solved;
		++n_steps)
	{
		try {

			solved = Step (training);

		} catch (const char *error) {

			printf ("%s\tstill %d steps to try.\n",
				error,
				n_maxIterations - n_steps);
		}

		if (n_steps && (n_steps % n_keepalive) == 0)
			printf ("Training Loss: %e\n", Loss ());
	}

	if (n_steps >= n_maxIterations)
		throw ("Exceeded Iterations");

	return true;
}

template<typename T> bool 
NNet_t<T>::Step (const DataSet_t * const training)
{
	int batch = training->N ();
	int sample;
	bool done;

	Start ();

	if (n_useSGD)
		batch = n_SGDn * batch;

	for (int i = 0; i < batch; ++i)
	{
		(n_useSGD ? sample = n_SGDsamples->SampleAuto () : sample = i);
		assert (sample >= 0 && sample < training->N ());
		ComputeDerivative (training->entry (sample));
	}

	done = Halt (training);
	if (!done)
		UpdateWeights ();

	return done;
}

