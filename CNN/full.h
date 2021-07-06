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

#ifndef __DJS_FULL__H__
#define __DJS_FULL__H__

#include <sys/uio.h>

#include <softmax.h>
#include <layer.h>

class full_t : public mapAPI_t
{
	Softmax_t		*re_net;

	int				re_Nin;

	double			*re_input;
	NeuralM_t		*re_gradient;

public:

	full_t (const int * const layers, const int Nlayers) :
		re_Nin (layers[0])
	{
		re_net = new Softmax_t (layers, Nlayers);
		re_input = new double [re_Nin + 1]; // + 1 for training - the answer
		re_gradient = new NeuralM_t (re_Nin, 1);
		ma_map.dd_rows = re_Nin;
		ma_map.dd_columns = 1;
		ma_map.dd_datap = re_gradient->raw ();
	}

	~full_t (void)
	{
		delete re_net;
		delete [] re_input;
		delete re_gradient;
	}

	bool Forward (arg_t &arg)
	{
		Load (arg);
		ma_signal = (int) re_net->Compute (re_input);

		return true;
	}

	int Load (arg_t &arg)
	{
		int blockSize = arg.a_args[0]->N ();
		int index = 0;

		for (int i = 0; i < arg.a_N; ++i, index += blockSize)
			memcpy (re_input + index, 
				arg.a_args[i]->raw (), 
				blockSize * sizeof (double));

		return index;
	}

	bool Train (arg_t &arg, double answer)
	{
		int len = Load (arg);

		assert (len == re_Nin);

		re_input[len] = answer;

		re_net->ComputeDerivative (re_input);
		re_net->ExposeGradient (*re_gradient);

		return true;
	}

	bool Backward (arg_t &arg)
	{
		return true;
	}

	bool Update (void)
	{
		re_net->UpdateWeights ();
		re_net->Start ();

		return true;
	}

	double Loss (void)
	{
printf ("DJS PROGRESS\t%f\t%f\n", re_net->Loss (), re_net->Accuracy ());
		return re_net->Error (NULL);
	}

	plane_t *fetchGradient (void)
	{
		return &ma_map;
	}
};

#endif // header inclusion

