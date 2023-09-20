/*

Copyright (c) 2023, Douglas Santry
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

#ifndef _NNm_PPLAYER__H__
#define _NNm_PPLAYER__H__

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include <NeuralM.h>
#include <NNm.h>

/*
 * Preprocessing Layer: translate and normalize data.
 *
 */
struct PPLayer_t : public stratum_t
{
	NeuralM_t				pl_Parameters;

	PPLayer_t (const int ID, DataSet_t *O) : 
		stratum_t ("PP", ID, O->Nin (), O->Nin ()),
		pl_Parameters (O->Nin (), 2)
	{
		IEEE_t *p = pl_Parameters.raw ();
		for (int i = 0; i < s_Nin; ++i)
		{
			p[0] = O->Mean (i);
			p[1] = O->StdDev (i);

			if (p[1] == 0)
				p[1] = 1.0; // useless division better than branching

			p += 2;
		}

		s_strat = NULL; // no learnable parameters
	}

	PPLayer_t (FILE *fp) : stratum_t ("PP")
	{
		Load (fp);
	}

	virtual ~PPLayer_t (void)
	{
	}

	void _sAPI_init (void)
	{
	}

	virtual int _sAPI_Store (FILE *fp) 
	{
		int bytes;

		bytes = fprintf (fp, "@PPLayer\n");
		if (bytes < 1)
			return errno;

		bytes = fprintf (fp, "@Dim\t%d\t%d\n", 
			pl_Parameters.rows (), 
			pl_Parameters.columns ());
		if (bytes < 1)
			return errno;

		pl_Parameters.displayExp ("@Weights", fp);

		return 0;
	}

	int Load (FILE *fp)
	{
		char buffer[MAXLAYERNAME];
		int rows, columns;
		int rc;

		rc = fscanf (fp, "%s %d %d\n", buffer, &rows, &columns);
		if (rc != 3)
			throw ("Invalid PPLayer_t dim");

		if (strcmp ("@Dim", buffer) != 0)
			throw ("PPLayer_t missing @Dim");

		if (rows < 1)
			throw ("invalid dense rows");

		if (columns < 1)
			throw ("invalid dense columns");

		s_Nnodes = rows;
		s_Nin = rows;
		s_response.resize (rows, 1);

		rc = fscanf (fp, "%s\n", buffer);
		if (rc != 1)
			throw ("No Weights");

		rc = pl_Parameters.Load (fp, rows, columns);
		if (rc)
			throw ("Bad Dense Weights");

		return 0;
	}

	virtual IEEE_t * _sAPI_f (IEEE_t * const, bool = true);
	virtual void _sAPI_gradient (stratum_t &);
	virtual void _sAPI_bprop (IEEE_t *, bool = true);

	virtual void StrategyMono (const int index)
	{
		return; // over-ride when debugging or instrumenting
	}
};

void
PPLayer_t::_sAPI_gradient (stratum_t &Z)
{
	Z.s_delta.Accept (s_delta.raw ());
}

void 
PPLayer_t::_sAPI_bprop (IEEE_t *xi, bool activation)
{
}

IEEE_t *
PPLayer_t::_sAPI_f (IEEE_t * const xi, bool activate)
{
	IEEE_t *p = pl_Parameters.raw ();
	IEEE_t *q = s_response.raw ();

	for (int i = 0; i < s_Nin; ++i, p += 2)
	{
		q[i] = xi[i] - p[0];
		q[i] = q[i] / p[1];
	}

	return s_response.sm_data;
}

#endif // header inclusion

