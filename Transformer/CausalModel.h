
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
	LanguageHead_t				m_L;

	Md_t						m_dX;

	counter_t					m_loss;
	counter_t					m_accuracy;

public:

	CausalModel_t (int N, int h, int l, int d, int V) :
		SparseCrossEntropy_t (),
		m_T (N, h, l, d, true),
		m_L (d, V)
	{
	}

	CausalModel_t (FILE *fp) :
		SparseCrossEntropy_t (),
		m_T (fp),
		m_L (fp)
	{
	}

	~CausalModel_t (void)
	{
	}

	int const * const predict (Md_t &X)
	{
		Md_t Z = m_T.call (X);
		(void) m_L.call (Z);

		return m_L.tokens ();
	}

	int const * const fit (Md_t &X, int const * const y)
	{
		int const * _y = predict (X);
		Md_t S = m_L.S ();

		Md_t &G = loss (S, y, _y);

		G = m_L.backward (G);
		m_dX = m_T.backward (G);

		return _y;
	}

	int const * const fit (exemplar_t &datum)
	{
		return fit (datum.first, datum.second);
	}

	IEEE_t fit (CausalData_t &K,
		int Nsamples,
		const int MaxEpochs=128, 
		const int batchSize=32,
		const bool verbose=false)
	{
		K.setMinLen (8);
//		K.setMaxLen (25);

		for (int epoch = 0; epoch < MaxEpochs; ++epoch)
		{
			m_loss.reset ();
			m_accuracy.reset ();

			for (int i = 1; i <= Nsamples;)
			{
				exemplar_t &y = K.getDatum ();
				if (y.second == NULL)
				{
					Nsamples = i + 1;
					break;
				}

				++i; // only increment i on accepted sequence

				(void) fit (y);

#ifndef __NO_E_LEARNING
				K.backward (m_dX, y.second);
#endif
				if ((i % batchSize) == 0)
				{
					if (verbose)
						printf ("BATCH %d/%d:\t%f\t%f\n",
							epoch,
							i / batchSize,
							getLoss (),
							getAccuracy ());
					else
						write (1, ".", 1);

					m_loss += getLoss ();
					m_accuracy += getAccuracy ();

					update ();
#ifndef __NO_E_LEARNING
					K.update ();
#endif
				}
			}

			if (getAccuracy () == 1.0)
			{
				printf ("\nEarly termination\n");
				break;
			}

			printf ("EPOCH %d:\t%f\t%f\n",
				epoch,
				m_loss.get (),
				m_accuracy.get ());

			K.reset ();
		}

		return m_loss.get ();
	}

	void update (void)
	{
		m_L.update ();
		m_T.update ();

		reset ();
	}

	bool save (char const * const filename)
	{
		FILE *fp = fopen (filename, "w");
		if (fp == NULL)
			return false;

		bool rc = fprintf (fp, "@CAUSALMODEL\n");

		rc &= m_T.save (fp);
		rc &= m_L.save (fp);

		fclose (fp);

		return rc;
	}

	int N_LearnableParameters (void) const
    {
        return m_T.N_LearnableParameters () + m_L.N_LearnableParameters ();
    }
};

#endif // header inclusion

