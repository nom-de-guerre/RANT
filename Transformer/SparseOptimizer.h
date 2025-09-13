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

#ifndef __RANT_SPARSE_ADAM__H__
#define __RANT_SPARSE_ADAM__H__

#include <unordered_map>
typedef std::unordered_map<int, int> RowMap_t;

#include <Optimizer.h>

/*
 * A batch size of 32 with a sequence length of 25 is 800, that is
 * 10% of a dictionary with |V| = 8,000 or so.  The embeddings are
 * |V| x d, and for d=512 that is 4,096,000 learnable parameters, 90%
 * of which do not need to be touched in a given update.
 * 
 * This class only updates the rows of X that were used in a batch.
 *
 */

struct SparseOptimizer_t : public Optimizer_t
{
	const int				so_columns;
	RowMap_t				so_used;

	SparseOptimizer_t (const int rows,
						const int columns,
						IEEE_t *X,
						IEEE_t *dX) :
		Optimizer_t (rows, columns, X, dX),
		so_columns (columns)
	{
	}

	~SparseOptimizer_t (void)
	{
	}

	void touch (const int row)
	{
		so_used[row] = row;
	}

	void update (void)
	{
		RowMap_t::iterator it;

		while ((it = so_used.begin ()) != so_used.end ())
		{
			int index = it->second;

			index *= so_columns;

			for (int i = 0; i < so_columns; ++i)
				ADAM (index + i);

			so_used.erase (it->first);
		}
	}
};

#endif // header inclusion

