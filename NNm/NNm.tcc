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

IEEE_t *
NNet_t::ComputeWork (IEEE_t *x)
{
	IEEE_t *ripple;

	ripple = x;

	for (int layer = 0; layer < n_populated; ++layer)
		ripple = n_strata[layer]->_sAPI_f (ripple);

	return ripple;
}

void
NNet_t::ComputeDerivative (const TrainingRow_t x)
{
	IEEE_t *Xi;

	/*
	 * Initiate the recurrence by triggering the loss function.
	 *
	 */
	ComputeWork (x);

	n_error += n_strata[n_populated - 1]->_sAPI_Loss (x + n_Nin);
	if (n_populated > 1)
		n_strata[n_populated - 1]->_sAPI_bprop (n_strata[n_populated-2]->z ());
	else
		n_strata[n_populated - 1]->_sAPI_bprop (x);

	for (int level = n_populated - 2; level >= 0; --level)
	{
		Xi = (level > 0 ? n_strata[level - 1]->z () : x);

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
		n_accuracy = 0;

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

	if (n_HaltOnAccuracy && n_accuracy == batch)
		done = true;

	done = n_error <= n_halt;
	if (!done)
		UpdateWeights ();

	return done;
}

void NNet_t::LoadModel (const char *filename)
{
	FILE *fp = fopen (filename, "r");
	int rc;

	if (fp == NULL)
		throw (strerror (errno));

	{
		float version;
		char buffer[512];

		rc = fscanf (fp, "%s %f\n", buffer, &version);
		if (rc != 2)
			throw ("Invalid version");

		if (strcmp ("@Version", buffer) != 0)
			throw ("Invalid version line");

		if (version != 1.0)
			throw ("Invalid version number");
	}

	{
		char buffer[512];

		rc = fscanf (fp, "%s %d %d %d\n",
			buffer,
			&n_Nin,
			&n_populated,
			&n_Nout);

		if (rc != 4)
			throw ("Invalid topology");

		if (strcmp ("@Topology", buffer) != 0)
			throw ("Invalid topology line");

		if (n_Nin < 1)
			throw ("Bad Nin");

		if (n_populated < 1)
			throw ("Bad number of layers");

		if (n_Nout < 1)
			throw ("Bad Nout");

		n_levels = n_populated;
		n_strata = new stratum_t * [n_populated];
	}

	bool working = true;

	do {

		char buffer[512];

		rc = fscanf (fp, "%s\n", buffer);
		if (rc != 1)
			throw ("Invalid NNet config");

		if (strcmp ("@Layers", buffer) == 0)
			working = false;
		else if (strcmp ("@Preprocess", buffer) == 0)
			LoadPreprocessing (fp);
		else
			throw ("Invalid NNet Cmd");

	} while (working);

	for (int i = 0; i < n_populated; ++i)
	{
		char lType[MAXLAYERNAME];

		rc = fscanf (fp, "%s\n", lType);
		if (rc != 1)
			throw ("Invalid NNm");

		if (strcmp (lType, "@Dense") == 0)
			n_strata[i] = new dense_t (fp);
		else if (strcmp (lType, "@MSE") == 0)
			n_strata[i] = new ScalerMSE_t (fp);
		else if (strcmp (lType, "@MLE") == 0)
			n_strata[i] = new SoftmaxMLE_t (fp, n_Nout);
		else
			throw ("Unknown Layer");
	}

	fclose (fp);
}

