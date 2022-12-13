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

#include <sys/stat.h>

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

#include <data.h>

#define STR_FEATURE 16

#define BUF_LEN 128

class LoadCSV_t
{
	FILE			*fs_fp;

	uint8_t			*fs_datap;
	int				fs_cursor;
	ssize_t			fs_free;
	ssize_t			fs_Bsize;

	char			**fs_titles;

	int				fs_rows;
	int				fs_columns;

	types_e			*fs_schema;

	void * Consume (ssize_t req)
	{
		void *p = fs_datap + fs_cursor;

		if (req > fs_free)
		{
			uint8_t *newSpace = new uint8_t [fs_Bsize * 2];

			memcpy (newSpace, fs_datap, fs_Bsize);
			fs_free += fs_Bsize;
			fs_Bsize <<= 1;

			delete [] fs_datap;

			fs_datap = newSpace;
			p = fs_datap + fs_cursor;
		}

		fs_free -= req;
		fs_cursor += req;

		return p;
	}

	types_e ProcessEntry (const char *buffer, Categories_t *dictp = NULL)
	{
		const char *p = buffer;
		int classID;
		bool is_numeric = true;
		bool dpoint = false;
		bool expon = false;

		if (*p == '-' || *p == '+')
			++p;

		for (; *p; ++p)
		{
			if (*p >= '0' && *p <= '9')
				continue;
			else if (*p == '.')
			{
				if (dpoint == false)
					dpoint = true;
				else {

					is_numeric = false;
					break; // not a number
				}
			} else if (*p == 'e') {

				if (expon == false)
					expon = true;
				else {

					is_numeric = false;
					break; // not a number
				}

				if (p[1] == '+' || p[1] == '-')
					++p;
			} else
				is_numeric = false;
		}

		if (is_numeric) // use dpoint if you care about integers
		{
			*((IEEE_t *) Consume (sizeof (IEEE_t))) = atof (buffer);

			return types_e::IEEE;

		} else if (dictp) {

			classID = dictp->Encode (buffer);
			*((IEEE_t *) Consume (sizeof (IEEE_t))) = classID;

			return types_e::CATEGORICAL;

		} else {

			int len = strlen (buffer);
			memcpy (Consume (len), buffer, len);

			return types_e::CATEGORICAL;
		}

		return types_e::IGNORE;
	}

	inline void ProcessBuffer (
		types_e id, 
		const char *buffer, 
		Categories_t &dict)
	{
		int classID;

		switch (id) {

		case IEEE:

			*((IEEE_t *) Consume (sizeof (IEEE_t))) = atof (buffer);

			break;

		case CATEGORICAL:

			classID = dict.Encode (buffer);
			*((IEEE_t *) Consume (sizeof (IEEE_t))) = classID;

			break;

			default:

			assert (false);
		}
	}

	int ReadHeader (bool process [])
	{
		char buffer[BUF_LEN];
		int index = 0;
		int entries;

		fs_titles = new char * [fs_columns];

		for (int i = 0; i < fs_columns - 1; ++i)
		{
			entries = fscanf (fs_fp, "%[^,],", buffer);
			if (entries != 1)
			{
				for (int j = 0; j < i; ++j)
					free (fs_titles[j]);

				delete [] fs_titles;

				return -1;
			}

			if (!process || process[i])
			{
				assert (index < fs_columns);

				fs_titles[index] = strdup (buffer);
				++index;
			}
		}

		// need to consume new line
		fscanf (fs_fp, "%s\n", buffer);
		fs_titles [index++] = strdup (buffer);

		return index;
	}

public:

	LoadCSV_t (const char *filepath) :
		fs_cursor (0),
		fs_titles (NULL),
		fs_rows (0),
		fs_columns (0),
		fs_schema (NULL)
	{
		struct stat mdata;

		int rc = stat (filepath, &mdata);
		if (rc)
			throw (strerror (errno));

		fs_Bsize = fs_free = mdata.st_size << 1;

		fs_fp = fopen (filepath, "r");
		if (fs_fp == NULL)
			throw (strerror (errno));

		fs_datap = new uint8_t [fs_free];
	}

	~LoadCSV_t (void)
	{
		fclose (fs_fp);

		for (int i = 0; fs_titles && i < fs_columns; ++i)
			free (fs_titles[i]);

		if (fs_titles)
			delete [] fs_titles;

		if (fs_schema)
			delete [] fs_schema;
	}

	char const * const ColumnName (int col) const
	{
		if (col < 0 || col > fs_columns)
			return NULL;

		return fs_titles[col];
	}

	void * Load (const int Nfeatures,
		int &rows,
		bool process[],
		bool header = true)
	{
		char buffer[BUF_LEN];
		int entries=1;

		if (header)
		{
			fs_columns = Nfeatures; // used in ReadHeader
			fs_columns = ReadHeader (process);
			if (fs_columns < 1)
				return NULL;

		} else {

			fs_columns = 0;
			for (int i = 0; i < Nfeatures; ++i)
				if (process[i])
					++fs_columns;
		}

		while (entries != EOF)
		{
			for (int i = 0; i < Nfeatures - 1; ++i)
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

	DataSet_t * LoadDS (const int Nfeatures,
		bool process[],
		bool header = true)
	{
		DataSet_t *O;
		Categories_t dict;
		char buffer[BUF_LEN];
		int entries=1;
		types_e type = IGNORE;
		int used;

		if (header)
		{
			fs_columns = Nfeatures;
			fs_columns = ReadHeader (process);
			if (fs_columns < 1)
				return NULL;

		} else {

			fs_columns = 0;
			for (int i = 0; i < Nfeatures; ++i)
				if (!process || process[i])
					++fs_columns;
		}

		fs_schema = new types_e [fs_columns];

		while (entries != EOF)
		{
			used = 0;
			for (int i = 0; i < Nfeatures - 1; ++i)
			{
				entries = fscanf (fs_fp, "%[^,],", buffer);
				if (entries != 1)
					break;

				if (process && !process[i])
					continue;

				type = ProcessEntry (buffer, &dict);
				if (fs_rows)
					assert (fs_schema[used] == type);
				else
					fs_schema[used] = type;

				++used;
			}

			entries = fscanf (fs_fp, ("%s\n"), buffer);
			if (entries != 1)
				break;

			if (!process || process[Nfeatures - 1])
			{
				ProcessEntry (buffer, &dict);
				if (fs_rows)
						assert (fs_schema[used] == type);
				else
						fs_schema[used] = type;
			}

			++fs_rows;
		}

		O = new DataSet_t (fs_rows, 
			fs_columns - 1, 
			1, 
			(IEEE_t *) fs_datap, 
			new ClassDict_t (dict));

		O->t_schema = fs_schema;
		fs_schema = NULL;

		return O;
	}

	DataSet_t * LoadSchema (const int Nfeatures,
		bool process [],
		types_e schema [],
		bool header = true)
	{
		DataSet_t *O = NULL;
		Categories_t dict;
		char buffer[BUF_LEN];
		int entries=1;

		if (header)
		{
			fs_columns = Nfeatures;
			fs_columns = ReadHeader (process);
			if (fs_columns < 1)
				return NULL;

		} else {

			fs_columns = 0;
			for (int i = 0; i < Nfeatures; ++i)
				if (process[i])
					++fs_columns;
		}

		while (entries != EOF)
		{
			for (int i = 0; i < Nfeatures - 1; ++i)
			{
				entries = fscanf (fs_fp, "%[^,],", buffer);

				if (!process[i])
					continue;

				ProcessBuffer (schema[i], buffer, dict);
			}

			// entries = fscanf (fs_fp, "%[^,]\n", buffer);
			entries = fscanf (fs_fp, ("%s\n"), buffer);
			ProcessBuffer (schema [Nfeatures - 1], buffer, dict);

			if (entries != EOF)
				++fs_rows;
		}

		O = new DataSet_t (fs_rows, 
			fs_columns - 1, 
			1, 
			(IEEE_t *) fs_datap, 
			new ClassDict_t (dict));

		return O;
	}

	int findIndex (char const * const feature)
	{
		for (int i = 0; fs_titles[i] != NULL; ++i)
			if (strcmp (feature, fs_titles[i]) == 0)
				return i;

		return -1;
	}
};

#undef BUF_LEN

#endif // header inclusion

