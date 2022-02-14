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

#ifndef __ANT_READ_CSV__H__
#define __ANT_READ_CSV__H__

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

#define MCHUNK	1048576 // 1 megabyte
#define STR_FEATURE 16

class LoadCSV_t
{
	FILE			*fs_fp;

	uint8_t			*fs_datap;
	int				fs_cursor;

	char			**fs_titles;

	int				fs_rows;
	int				fs_columns;

public:

	LoadCSV_t (const char *filepath) :
		fs_cursor (0),
		fs_titles (NULL),
		fs_rows (0),
		fs_columns (0)
	{
		extern int errno;

		fs_fp = fopen (filepath, "r");
		if (fs_fp == NULL)
			throw (strerror (errno));

		fs_datap = new uint8_t [MCHUNK];
	}

	~LoadCSV_t (void)
	{
		fclose (fs_fp);

		for (int i = 0; fs_titles && i < fs_columns; ++i)
			free (fs_titles[i]);

		if (fs_titles)
			delete [] fs_titles;
	}

	void * Load (const int Nfeatures,
		int &rows,
		bool process[],
		bool header = true)
	{
		char buffer[64];
		int entries;

		fs_columns = Nfeatures;

		if (header)
			fs_titles = new char * [Nfeatures];

		for (int i = 0; header && i < Nfeatures - 1; ++i)
		{
			entries = fscanf (fs_fp, "%[^,],", buffer);
			if (entries != 1)
			{
				for (int j = 0; j < i; ++j)
					free (fs_titles[j]);

				delete [] fs_titles;

				return NULL;
			}

			fs_titles[i] = strdup (buffer);
		}

		if (header) // need to consume new line
		{
			fscanf (fs_fp, "%s\n", buffer);
			fs_titles [Nfeatures - 1] = strdup (buffer);
		}

		while (entries != EOF)
		{
			for (int i = 0; header && i < Nfeatures - 1; ++i)
			{
				entries = fscanf (fs_fp, "%[^,],", buffer);
				if (entries != 1)
					break;

				if (process && process[i])
					ProcessEntry (buffer);
			}

			entries = fscanf (fs_fp, ("%s\n"), buffer);
			if (process && process[Nfeatures - 1])
				ProcessEntry (buffer);

			if (entries == 1)
				++fs_rows;
		}

		rows = fs_rows;

		return fs_datap;
	}

	void ProcessEntry (const char *buffer)
	{
		const char *p = buffer;
		bool is_numeric = true;
		bool is_float = false;

		while (*p)
		{
			if (*p < '0' || *p > '9')
			{
				is_numeric = false;
				if (is_float) {

					is_float = false;
					break;

				} else if (*p == '.') {
					is_float = true;
				} else
					break; // treat as string
			}

			++p;
		}

		if (is_numeric || is_float)
		{
			*(double *) (fs_datap + fs_cursor) = atof (buffer);
			fs_cursor += sizeof (double);

		} else {

			int len = strlen (buffer);
			memcpy (fs_datap + fs_cursor, buffer, len);
			fs_cursor += STR_FEATURE;
		}

		assert (fs_cursor < MCHUNK);
	}
};

#endif // header inclusion

