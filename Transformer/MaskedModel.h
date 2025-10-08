
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

#ifndef __RANT_MASKED_LANGUAGE_MODEL__H__
#define __RANT_MASKED_LANGUAGE_MODEL__H__

#include <transformer.h>
#include <LanguageHead.h>
#include <loss.h>

class MaskedModel_t : public SparseCrossEntropy_t
{
	transformer_t				m_T;
	LanguageHead_t				m_L;

	Md_t						m_dX;

	counter_t					m_loss;
	counter_t					m_accuracy;

public:

	MaskedModel_t (int N, int h, int l, int d, int V) :
		SparseCrossEntropy_t (),
		m_T (N, h, l, d),
		m_L (d, V)
	{
	}

	~MaskedModel_t (void)
	{
	}

	int const * const predict (Md_t &X)
	{
		Md_t Z = m_T.call (X);
		(void) m_L.call (Z);

		return m_L.tokens ();
	}

	int const * const fit_work (Md_t &X, int const * const y)
	{
		int const * _y = predict (X);
		Md_t S = m_L.S ();

		Md_t &G_full = loss (S, y, _y);

		Md_t G (G_full.rows (), G_full.columns (), 0.0);

		// filter gradient for masked tokens only
		int rows = G_full.rows ();
		for (int i = 0; i < rows; ++i)
		{
			if (y[i] < 0) // negative tokens are masked
				continue;

			G.importRow (i, G_full);
		}

		G = m_L.backward (G);
		m_dX = m_T.backward (G);

		return _y;
	}

	int const * const fit (exemplar_t &datum)
	{
		return fit_work (datum.first, datum.second);
	}

	void update (void)
	{
		m_L.update ();
		m_T.update ();

		reset ();
	}

	IEEE_t fit (MaskedData_t &K,
		int Nsamples,
		const int MaxEpochs=128, 
		const int batchSize=32,
		const bool verbose=false)
	{
		K.setMinLen (5);

		for (int epoch = 0; epoch < MaxEpochs; ++epoch)
		{
			m_loss.reset ();
			m_accuracy.reset ();

			for (int i = 1; i <= Nsamples; ++i)
			{
				exemplar_t &datum = K.getDatum ();
				if (datum.second == NULL)
				{
					Nsamples = i;
					break; // bail on batch
				}

				assert (datum.first.isRowOrder ());

#ifdef __SHOW_PREDICTIONS
				int len = datum.first.rows ();
				int const * const _y = fit (datum);
#else
				(void) fit (datum);
#endif
#ifndef __NO_E_LEARNING
				K.backward (m_dX, datum.second);
#endif

				if ((i % batchSize) == 0)
				{
					if (verbose)
						printf ("BATCH %d:\t%f\t%f\n",
							i / batchSize,
							getLoss (),
							getAccuracy ());
					else
						write (1, ".", 1);

					m_loss += getLoss (),
					m_accuracy += getAccuracy ();

					update ();
#ifndef __NO_E_LEARNING
					K.update ();
#endif
				}

#ifdef __SHOW_PREDICTIONS
				if (epoch < 13)
					continue;

				int const * const y = datum.second;

				for (int m = 0; m < len; ++m)
				{
					if (y[m] == __MASKED_SKIP_POSITION)
						continue;

					printf ("%d\t%d:\t%s\t%s\n", 
						epoch, 
						i, 
						K.cd_V.TokenToString (y[m]),
						K.cd_V.TokenToString (_y[m]));
				}
#endif
			}

//			if (verbose)
				printf ("EPOCH %d:\t%f\t%f\n",
					epoch,
					m_loss.get (),
					m_accuracy.get ());

			if (m_accuracy.get () == 1.0)
				break;

			K.reset ();
		}

		return m_loss.get ();
	}

	int N_LearnableParameters (void) const
	{
		return m_T.N_LearnableParameters () + m_L.N_LearnableParameters ();
	}
};

#endif // header inclusion

