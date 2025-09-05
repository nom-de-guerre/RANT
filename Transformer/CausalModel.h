
/*  
 
Copyright (c) 2025, Douglas Santry 
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

#ifndef __RANT_CAUSAL_LANGUAGE_MODEL__H__
#define __RANT_CAUSAL_LANGUAGE_MODEL__H__

#include <transformer.h>
#include <LanguageHead.h>
#include <loss.h>

class CausalModel_t : public SparseCrossEntropy_t
{
	transformer_t				m_T;
	LanguageHead_t				m_V;

	Md_t						m_dX;

	counter_t					m_loss;
	counter_t					m_accuracy;

public:

	CausalModel_t (int N, int h, int l, int d, int V) :
		SparseCrossEntropy_t (),
		m_T (N, h, l, d, true),
		m_V (d, V)
	{
	}

	~CausalModel_t (void)
	{
	}

	int const * const predict (Md_t &X)
	{
		Md_t Z = m_T.call (X);
		(void) m_V.call (Z);

		return m_V.tokens ();
	}

	int const * const fit (Md_t &X, int const * const y)
	{
		int const * _y = predict (X);
		Md_t S = m_V.S ();

		Md_t &G = loss (S, y, _y);

		G = m_V.backward (G);
		m_dX = m_T.backward (G);

		return _y;
	}

	int const * const fit (exemplar_t &datum)
	{
		return fit (datum.first, datum.second);
	}

	void update (void)
	{
		m_V.update ();
		m_T.update ();

		reset ();
	}

	IEEE_t fit (CausalData_t &K,
		int Nsamples,
		const int MaxEpochs=128, 
		const int batchSize=32,
		const bool verbose=false)
	{
		for (int epoch = 0; epoch < MaxEpochs; ++epoch)
		{
			m_loss.reset ();
			m_accuracy.reset ();

			for (int i = 0; i < Nsamples;)
			{
				exemplar_t &y = K.getDatum ();
				if (y.second == NULL)
				{
					Nsamples = i + 1;
					break;
				}

				int len = y.first.rows ();

				if (len < 5 || len > 25)
					continue;

				++i; // only increment i on accepted sequence

				(void) fit (y);

				if ((i % batchSize) == 0)
				{
					if (verbose)
						printf ("BATCH %d/%d:\t%f\t%f\n",
							epoch,
							i / batchSize,
							getLoss (),
							getAccuracy ());

					m_loss += getLoss ();
					m_accuracy += getAccuracy ();

					update ();
				}
			}

			if (getAccuracy () == 1.0)
				break;

			printf ("EPOCH %d:\t%f\t%f\n",
				epoch,
				m_loss.get (),
				m_accuracy.get ());

			K.reset ();
		}

		return m_loss.get ();
	}
};

#endif // header inclusion

