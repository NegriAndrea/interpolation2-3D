#!/usr/bin/env python
# -*- coding: utf-8 -*-

def chopArray(arraySize, Nchunks):
    """
    Code taken from array_split, except that it takes the size of an array, and
    returns the offsets and dimensions of the subarrays, divided in Nchunks of
    approximately equal size.

    arraySize: size of the array to chop
    Nchunks : number of chunks to dive the array in

    Works on 1D array only.

    Returns:

    offsets : (Nchunks,) integer array
    sizes   : (Nchunks,) integer array

    """
    import numpy as np

    # N is a scalar
    Nsections = int(Nchunks)

    if Nsections <= 0:
        raise ValueError('N must be larger than 0.')
    Neach_section, extras = divmod(arraySize, Nsections)
    sizes = (extras * [Neach_section+1] +
                     (Nsections-extras) * [Neach_section])

    offsets = np.array([0]+sizes[:-1], dtype=np.intp).cumsum()
    sizes = np.asarray(sizes, dtype=np.intp)

    assert offsets.shape == (Nchunks,)
    assert sizes.shape == (Nchunks,)
    assert offsets[-1]+sizes[-1] == arraySize

    return offsets, sizes
