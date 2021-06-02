#ifndef __DJS_MAXPOOL__H__
#define __DJS_MAXPOOL__H__

#include <layer.h>

class Mpool_t : public mapAPI_t
{
	int				mp_width;		// only square filters currently supported
	plane_t			mp_grad;		// gradient
	plane_t			mp_rindex;		// reverse index, source of max

public:

	/*
	 * We need the pool width and the input map width.
	 *
	 */
	Mpool_t (const int mwidth, const int iwidth) : 
		mapAPI_t (iwidth - mwidth + 1),
		mp_width (mwidth),
		mp_grad (iwidth, iwidth),
		mp_rindex (iwidth, iwidth)
	{
	}

	~Mpool_t (void)
	{
	}

	/*
	 * mapAPI_t interface
	 *
	 */
	bool Forward (arg_t &arg)
	{
		return Pool (arg.a_args[0]);
	}

	bool Train (arg_t &arg, double answer)
	{
		memset (mp_grad.raw (), 0, mp_grad.N () * sizeof (double));
		Pool (arg.a_args[0]);

		return true;
	}

	bool Backward (arg_t &arg)
	{
		assert (arg.a_N == 1);

		ComputeGradient (arg.a_args[0]);

		return true;
	}

	bool Update (void)
	{
		return true;
	}

	plane_t *fetchGradient (void)
	{
		return &mp_grad;
	}

	/*
	 * class specific helper functions.
	 *
	 */
	bool Pool (plane_t const * const datap);
	bool ComputeGradient (plane_t const * const datap);
};

bool Mpool_t::Pool (plane_t const * const datap)
{
	__restrict double *omap = ma_map.raw ();
	__restrict double *rindexp = mp_rindex.raw ();
	__restrict double *imagep = datap->raw ();

	int idim = datap->rows ();
	int mdim = ma_map.rows ();
	int stride = idim - mp_width;

	for (int start = 0, index = 0, i = 0; i < mdim; ++i, start = i * idim)
		for (int i_idx = 0, j = 0; j < mdim; ++j, ++index, ++start)
		{
			omap[index] = -DBL_MAX;
			i_idx = start;

			for (int k = 0; k < mp_width; ++k, i_idx += stride)
				for (int l = 0; l < mp_width; ++l, ++i_idx)
					if (omap[index] < imagep[i_idx])
					{
						omap[index] = imagep[i_idx];
						rindexp[index] = i_idx;
					}
		}

#ifdef MAX_DEBUG
	ma_map.display ("Mpool");
	for (int i = 0; i < MapSize () ; ++i)
{
		printf ("%d  ", (int) rindexp[i]);
assert (rindexp[i] < 676);
}
	printf ("\n");
#endif

	return true;
}

bool Mpool_t::ComputeGradient (plane_t const * const datap)
{
	__restrict double *gradp = mp_grad.raw ();
	__restrict double *rindexp = mp_rindex.raw ();
	__restrict double *deltap = datap->raw ();
	int halt = mp_grad.N ();

	for (int i = 0; i < halt; ++i)
		gradp[(int) rindexp[i]] += deltap[i];

	return true;
}

#endif // header inclusion

