
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

#ifndef __RANT_LEXEMES__H__
#define __RANT_LEXEMES__H__

#include <sys/uio.h>

#include <ctype.h>
#include <string.h>

#include <mfile.h>

char lexeme_page[4096];

class TextDocument_t : mapFile_t
{
	char const *			td_cursor;
	int						td_consumed;

	int						td_lexemes;
	int						td_clauses;

public:

	TextDocument_t (char const * const filename) :
		mapFile_t (filename)
	{
		Reset ();
	}

	void Reset (void)
	{
		td_cursor = (char const *) mf_base;
		td_consumed = 0;

		td_lexemes = 0;
		td_clauses = 0;
	}

	int GetLexemesScanned (void) const
	{
		return td_lexemes;
	}

	int GetNumberSentences (void) const
	{
		return td_clauses;
	}

	int NextClause (const int Max, struct iovec *tokens)
	{
		int len = 1;
		int N = 0;
		char *base = lexeme_page;
		char const *p;

		while (len > 0 && N < Max)
		{
			p = parse (len);
			if (len > 0)
			{
//				printf ("Found %.*s\n", len, p);

				strncpy (base, p, len);
				base[len] = 0;
				tokens[N].iov_base = base;
				tokens[N].iov_len = len;

				if (strncmp (base, "qzmSEPqzm", 9) == 0)
					return N + 1;

				base += len + 1;

			} else
				return -1;

			++N;
		}

		return N - 1;
	}

	/*
	 * Return 1 for found a lexeme
	 * Return -1 for EOF
	 *
	 */
	char const *parse (int &len)
	{

		while (td_consumed < mf_len && !isalpha (*td_cursor))
		{
			++td_consumed;
			++td_cursor;
		}

		if (td_consumed == mf_len)
		{
			len = -1;
			return NULL;
		}

		char const *p = td_cursor;
		len = 0;

		while (td_consumed < mf_len && isalpha (*td_cursor))
		{
			++td_consumed;
			++td_cursor;
			++len;
		}

		return p;
	}

	int __NextClause (const int Max, struct iovec *tokens)
	{
		int N = 0;
		bool alpha = false;
		char *base = lexeme_page;
		char *target = base;

		if (td_consumed == mf_len)
			return -1;

		// Fast forward to a lexeme
		while (td_consumed < mf_len)
		{
			char ascii = *td_cursor;

			if (isalpha (ascii) || isdigit (ascii))
				break;

			++td_cursor;
			++td_consumed;
		}

		tokens[0].iov_base = (void *) td_cursor;
		tokens[0].iov_len = 0;

		// Consume a sentence
		while (td_consumed < mf_len)
		{
			char ascii = *td_cursor;

			if (isalpha (ascii))
			{
				if (isupper (ascii))
					*target = tolower (ascii);
				else
					*target = ascii;

				++target;
				++tokens[N].iov_len;
				alpha = true;

 			} else if (isdigit (ascii)) {

				*target = ascii;

				++target;
				++tokens[N].iov_len;

			} else {

				/*
				 * Accept lexeme and consume white space.
				 *
				 */

				if (alpha)
				{
					alpha = false;

					*target = 0;
					tokens[N].iov_base = base;

					++target;
					base = target;

					++td_lexemes;
					++N;
					if (N == Max)
						return -1;

				} else
					target = base;

				while (!isalpha (ascii) && !isdigit (ascii))
				{
					if (ascii == '.' || 
//						ascii == ',' || 
						ascii == ';' || 
						ascii == ':')
					{
						++td_clauses;

						return N;
					}

					++td_cursor;
					++td_consumed;
					if (td_consumed == mf_len)
					{
						++td_clauses;

						return N;
					}

					ascii = *td_cursor;
				}

				tokens[N].iov_base = NULL;
				tokens[N].iov_len = 0;

				continue;
			}

			++td_cursor;
			++td_consumed;
		}

		return N;
	}
};

#endif // header inclusion

