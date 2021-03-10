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

#ifndef __MFILE__H__
#define __MFILE__H__

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>
#include <stdint.h>

#include <string.h>

struct mapFile_t 
{
	int				mf_fd;
	size_t			mf_len;
	void			*mf_base;

	mapFile_t (const char * filename)
	{
		struct stat meta;
		int rc;

		mf_fd = open (filename, O_RDONLY);
		if (mf_fd < 0)
			throw (strerror (errno));

		rc = fstat (mf_fd, &meta);
		if (rc)
		{
			close (mf_fd);
			throw (strerror (errno));
		}

		mf_len = meta.st_size;

		mf_base = mmap (NULL,
				mf_len,
				PROT_READ,
				MAP_SHARED | MAP_FILE,
				mf_fd,
				0ll);
		if (mf_base == (void *) -1)
		{
			close (mf_fd);
			throw (strerror (errno));
		}
	}

	~mapFile_t (void)
	{
		munmap (mf_base, mf_len);
		close (mf_fd);
	}
};

#endif // header inclusion

