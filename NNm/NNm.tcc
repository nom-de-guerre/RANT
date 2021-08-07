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
	return static_cast<T *> (this)->Cycle ();
}

template<typename T> bool
NNet_t<T>::Halt (DataSet_t const * const tp)
{
	// Finished a batch, can we stop?
	return static_cast<T *> (this)->Test (tp);
}

template<typename T> double
NNet_t<T>::Loss (DataSet_t const *tp)
{
	// The current value of the loss function
	return static_cast<T *> (this)->Error (tp);
}

template<typename T> double 
NNet_t<T>::ComputeDerivative (const TrainingRow_t x)
{
	/*
	 * Initiate the recurrence by triggering the loss function.
	 *
	 */
	double error = static_cast<T *>(this)->bprop (x);

	for (int level = n_levels - 2; level >= 0; --level)
		n_strata[level]->bprop (
			*n_strata[level + 1],
			(level > 0 ? n_strata[level - 1]->s_response.raw () : x));

	return error;
}

template<typename T> double
NNet_t<T>::Compute (double *x)
{
	double *ripple;

	ripple = x;

	for (int layer = 0; layer < n_levels - 1; ++layer)
		ripple = n_strata[layer]->f (ripple);

	/*
	 * Compute the final result with the specialization
	 *
	 */
	return static_cast<T *>(this)->f (ripple);
}

template<typename T> bool 
NNet_t<T>::Train (const DataSet_t * const training, int maxIterations)
{
	bool rc;

	rc = TrainWork (training, maxIterations);

	return rc;
}

template<typename T> bool 
NNet_t<T>::TrainWork (const DataSet_t * const training, int maxIterations)
{
	bool solved = false;

	for (n_steps = 0; 
		(n_steps < maxIterations) && !solved; 
		++n_steps)
	{
		try {

			solved = Step (training);

		} catch (const char *error) {

			printf ("%s\tstill %d steps to try.\n",
				error,
				maxIterations - n_steps);
		}

		if ((n_steps % 10000) == 0)
			printf ("Loss: %e\n", n_error);
	}

	if (n_steps >= maxIterations)
		throw ("Exceeded Iterations");

	printf ("Finished training: %d\t%e\n", 
		n_steps, 
		n_error);

	return true;
}

template<typename T> bool 
NNet_t<T>::Step (const DataSet_t * const training)
{
	Start ();

	for (int i = 0; i < training->t_N; ++i)
		ComputeDerivative (training->entry (i));

	UpdateWeights ();

	return Halt (training);
}

/*
 * The below are used when a stratum is stand-alone trained (e.g. a filter).
 *
 */

template<typename T> bool
NNet_t<T>::ExposeGradient (NeuralM_t &grad)
{
	// Compute per node total derivative
	grad.TransposeMatrixVectorMult (
		n_strata[0]->s_W, 
		n_strata[0]->s_delta.raw ());

	return true;
}

