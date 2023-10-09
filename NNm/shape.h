/*

Copyright (c) 2022, Douglas Santry
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

#ifndef _NN_SHAPE__H__
#define _NN_SHAPE__H__

struct shape_t
{
	int			sh_N;
	int			sh_rows;
	int			sh_columns;
	int			sh_length;

	shape_t (const shape_t & X) :
		sh_N (X.sh_N),
		sh_rows (X.sh_rows),
		sh_columns (X.sh_columns),
		sh_length (X.sh_length)
	{
	}

	shape_t (int entries) :
		sh_N (1),
		sh_rows (entries),
		sh_columns (1),
		sh_length (entries)
	{
	}

	shape_t (int rows, int columns) :
		sh_N (1),
		sh_rows (rows),
		sh_columns (columns),
		sh_length (rows * columns)
	{
	}

	shape_t (int N, int rows, int columns) :
		sh_N (N),
		sh_rows (rows),
		sh_columns (columns),
		sh_length (N * rows * columns)
	{
	}

	shape_t (void)
	{
		sh_N = sh_rows = sh_columns = sh_length = -1;
	}

	int mapSize (void) const
	{
		return sh_rows * sh_columns;
	}

	int len (void) const
	{
		return sh_length;
	}

	void Display (const char *title = NULL, FILE *fp=stdout) const
	{
		fprintf (fp, "%s\t%d, %d, %d\n",
			(title ? title : "@Shape"),
			sh_N,
			sh_rows,
			sh_columns);
	}

	int N (void) const
	{
		return sh_N;
	}

	bool Single (void) const
	{
		return (sh_N == 1 ? true : false);
	}

	bool isFlat (void) const
	{	
		return (sh_columns == 1 ? true : false);
	}

	bool isSingle (void) const
	{
		return sh_N == 1;
	}

	void Load (FILE *fp)
	{
		char buffer[MAXLAYERNAME];
		int rc;

		rc = fscanf (fp, "%s %d, %d, %d\n",
			buffer,
			&sh_N,
			&sh_rows,
			&sh_columns);

		if (rc != 4)
			throw ("shape_t: invalid stored state");

		sh_length = sh_N * sh_rows * sh_columns;
	}

	int rows (void) const
	{
		return sh_rows;
	}

	int columns (void) const
	{
		return sh_columns;
	}
};

#endif // header inclusion

