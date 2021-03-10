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

void 
stratum_t::bprop (stratum_t &next, double *yi)
{
	/*
	 * Compute per node total derivative for node at layer i - 1.
	 *
	 * ∂L     ∂L   ∂wi
	 * --   = --   ---
	 * ∂y     ∂wi  ∂y
	 *
	 */
	s_delta.TransposeMatrixVectorMult (next.s_W, next.s_delta.raw ());

	// Compute per node delta
	for (int i = 0; i < s_Nperceptrons; ++i)
		s_delta.sm_data[i] *= DERIVATIVE_FN (s_response.sm_data[i]);

	// Apply the delta for per weight derivatives

	double *dL = s_dL.sm_data;
	double delta;

	/*
	 * ∂L   ∂y  ∂∑
	 * -- = --  --
	 * ∂w   ∂∑  ∂w
	 *
	 */
	for (int i = 0; i < s_Nperceptrons; ++i)
	{
		delta = s_delta.sm_data[i];
		*dL++ += delta; // the Bias

		for (int j = 1; j < s_Nin; ++j)
			*dL++ += delta * yi[j - 1];
	}
}

void 
stratum_t::bprop (double *yi)
{
	// Compute per node delta
	for (int i = 0; i < s_Nperceptrons; ++i)
		s_delta.sm_data[i] *= DERIVATIVE_FN (s_response.sm_data[i]);

	// Apply the delta for per weight derivatives

	double *dL = s_dL.sm_data;
	double delta;

	for (int i = 0; i < s_Nperceptrons; ++i)
	{
		delta = s_delta.sm_data[i];
		*dL++ += delta; // the Bias

		for (int j = 1; j < s_Nin; ++j)
			*dL++ += delta * yi[j - 1];
	}
}

void 
stratum_t::RPROP (void)
{
	int Nweights = s_Nperceptrons * s_Nin;

	for (int index = 0; index < Nweights; ++index)
		RPROP (index);

	s_dL.zero ();
}

void 
stratum_t::RPROP (int index)
{
	double delta;
	double backtrack;

	if (s_Ei.sm_data[index] == 0.0 || s_dL.sm_data[index] == 0.0)
	{
		delta = -SIGN (s_dL.sm_data[index]) * s_deltaW.sm_data[index];
		s_W.sm_data[index] += delta;

		s_Ei.sm_data[index] = s_dL.sm_data[index];

	} else if (signbit (s_dL.sm_data[index]) == signbit (s_Ei.sm_data[index])) {

		// (1)
		delta = s_deltaW.sm_data[index] * ETA_PLUS;
		if (delta > DELTA_MAX)
			delta = DELTA_MAX;

		s_deltaW.sm_data[index] = delta;

		// (2)
		delta *= -(SIGN (s_dL.sm_data[index]));
		// (3)
		s_W.sm_data[index] += delta;

		s_Ei.sm_data[index] = s_dL.sm_data[index];

	} else {

		backtrack = s_deltaW.sm_data[index] * SIGN (s_Ei.sm_data[index]);

		// (1)
		delta = s_deltaW.sm_data[index] * ETA_MINUS;
		if (delta < DELTA_MIN)
			delta = DELTA_MIN;

		s_deltaW.sm_data[index] = delta;

		// (2)
		s_W.sm_data[index] += backtrack;

		// (3)
		s_Ei.sm_data[index] = 0.0;
	}

	s_dL.sm_data[index] = 0.0;

	assert (s_deltaW.sm_data[index] > 0);
}

double *
stratum_t::f (double *xi)
{
	s_dot.MatrixVectorMult (s_W, xi);

	double *p = s_response.sm_data;
	double *dot = s_dot.sm_data;
	for (int i = 0; i < s_Nperceptrons; ++i)
		*p++ = ACTIVATION_FN (*dot++);

	return s_response.sm_data;
}

/*
 * Used when a stratum is stand-alone trained.
 *
 */
void
stratum_t::f (double *xi, double *result)
{
	s_dot.MatrixVectorMult (s_W, xi);

	double *p = s_response.sm_data;
	double *dot = s_dot.sm_data;
	for (int i = 0; i < s_Nperceptrons; ++i)
		*p++ = result[i] = ACTIVATION_FN (*dot++);

	return s_response.sm_data;
}

template<typename T> void
NNet_t<T>::PresentExamples (
	const DataSet_t * const training, 
	bool ComputeDerivatives)
{
	for (int i = 0; i < training->t_N; ++i)
	{
		const TrainingRow_t p = (*training)[i];
		ComputeDerivative (p, i);
	}
}

/*
 * Interface to loss function.
 *
 */
template<typename T> void 
NNet_t<T>::Start (void)
{
	n_error = 0.0;

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
	return static_cast<T *> (this)->error (tp);
}

/*
 * ∂E     ∂E  ∂ak  ∂net
 * --   = --  ---  ----
 * ∂wkj   ∂ak ∂net ∂wkj
 *
 * y = ak
 *
 * ∆k = (ak - t)ak(1 - ak)
 * wkj = ∆k(aj)
 *
 */

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

// printf ("theta = %f\n", x[0]);

	ripple = x;

	for (int layer = 0; layer < n_levels; ++layer)
		ripple = n_strata[layer]->f (ripple);

	/*
	 * Compute the final result with the specialization
	 *
	 */
	return static_cast<T *>(this)->f (ripple);
}

template<typename T> bool 
NNet_t<T>::Step (const DataSet_t * const training, double &progress)
{
	progress = 0;

	Start ();

	for (int i = 0; i < training->t_N; ++i)
		n_loss[i] = ComputeDerivative (training->entry (i));

	UpdateWeights ();

	return Halt (training);
}

template<typename T> bool 
NNet_t<T>::Train (const DataSet_t * const training, int maxIterations)
{
	bool rc;
	n_loss = new double [training->t_N];

	rc = TrainWork (training, maxIterations);

	delete [] n_loss;
	n_loss = NULL;

	return rc;
}

template<typename T> bool 
NNet_t<T>::TrainWork (const DataSet_t * const training, int maxIterations)
{
	bool solved = false;
	double progress = 1;

	n_error = 1.0;

	for (n_steps = 0; 
		(n_steps < maxIterations) && !solved; 
		++n_steps)
	{
		try {

			solved = Step (training, progress);

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
NNet_t<T>::ExposeGradient (NeuralM_t &grad)
{
	// Compute per node total derivative
	grad.TransposeMatrixVectorMult (
		n_strata[0]->s_W, 
		n_strata[0]->s_delta.raw ());

	return true;
}

