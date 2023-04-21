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

#ifndef __MNIST__H__
#define __MNIST__H__

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>

#include <string.h>

#include <data.h>
#include <mfile.h>

#define IMAGEDIM		28
#define IMAGEBYTES		(IMAGEDIM * IMAGEDIM)

struct MNIST_t
{
	DataSet_t			*mn_datap;

	MNIST_t (const char *datafile, const char *labelfile, bool PP = false)
	{
		mapFile_t data (datafile);
		mapFile_t labels (labelfile);
		int32_t *magic;

		magic = (int32_t *) labels.mf_base;
		assert (ntohl (*magic) == 2049);

		magic = (int32_t *) data.mf_base;
		assert (ntohl (*magic) == 2051);	// magic number
		assert (ntohl (magic[2]) == IMAGEDIM);	// # rows
		assert (ntohl (magic[3]) == IMAGEDIM);	// # columns

		int N_data = data.mf_len;
		N_data -= 4 * 4;
		N_data /= IMAGEBYTES;
		assert (ntohl (magic[1]) == N_data);

 		mn_datap = new DataSet_t (N_data, IMAGEBYTES, 1);
		uint8_t *image = (uint8_t *) (magic + 4);

		for (int i = 0; i < N_data; ++i)
		{
			for (int j = 0; j < IMAGEBYTES; ++j)
			{
				(*mn_datap)[i][j] = image[j];
				if (!PP)
					continue;

				(*mn_datap)[i][j] /= 255;
#if 0
				if ((*mn_datap)[i][j] < 0.35)
					(*mn_datap)[i][j] = 0.0;
#endif

				(*mn_datap)[i][j] -= 0.5;
			}

			image += IMAGEBYTES;
		}

		magic = (int32_t *) labels.mf_base;
		image = (uint8_t *) (magic + 2);

		for (int i = 0; i < N_data; ++i)
			(*mn_datap)[i][IMAGEBYTES] = image[i];
	}

	~MNIST_t (void)
	{
		delete mn_datap;
	}

	int N (void) const
	{
		return mn_datap->N ();
	}

	DataSet_t * Data (void)
	{
		return mn_datap;
	}
};

#endif // header inclusion

