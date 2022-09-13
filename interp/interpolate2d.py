#!/usr/bin/env python
# -*- coding: utf-8 -*-
__all__ = ['interpolate2_3D', 'init', 'interpolate_call']

import numpy as np
import time
from numba import njit, jit
from multiprocessing import Array
from .get_project_root import get_project_root
path_lib = str(get_project_root() / 'loopC.so')

#  ----------------------------------------------------------------------------------------
# note that this part is used with numba, since it's the only way I have found
# to send arrays to C code that work in numba, since np.ctypeslib is not
# supported in numba
from cffi import FFI
ffi = FFI()
C = ffi.dlopen(path_lib)
ffi.cdef('void loop(const size_t d0, const size_t d1, const size_t d2,'
    'const size_t npoints, const double values[], const double t[],  const double u[],'
    'const ssize_t ind0[], const ssize_t ind1[], double out[]);')
ffi.cdef('void loopfirstIndex(const size_t n, const size_t npoints,'
    'const size_t m, const size_t l, const size_t d0, const size_t d1,'
    'const size_t d2,  const double values[], const double u[],'
    'const ssize_t firstIndex[], const ssize_t ind1[], double output[]);')
ffi.cdef('void loopsecondIndex(const size_t n, const size_t npoints,'
    'const size_t m, const size_t l, const size_t d0, const size_t d1,'
    'const size_t d2,  const double values[], const double t[],'
    'const ssize_t ind0[], const ssize_t secondIndex[], double output[]);')
loop_ffi = C.loop
loopfirstindex_ffi = C.loopfirstIndex
loopsecondindex_ffi = C.loopsecondIndex
#  ----------------------------------------------------------------------------------------

def convert_bool(val):
    if type(val) is not bool:
        raise ValueError('only bools are accepted')
    out = np.array([val])
    nbytes = out.nbytes

    out2 = np.ctypeslib.as_array(Array('b', nbytes, lock=False)).view(bool)
    assert out2.nbytes == nbytes
    out2 [0] = val
    del out

    return out2


class interpolate2_3D(object):

    def __init__(self, points0, points1, values, method="linear", bounds_error=True,
                 fill_value=np.nan, copyInput = True):

        if method not in ["linear"]:
            raise ValueError("Method '%s' is not defined" % method)

        self.bounds_error = convert_bool(bounds_error)

        if len(points0.shape) != 1:
            raise ValueError('points0 must be 1d array')

        if len(points1.shape) != 1:
            raise ValueError('points1 must be 1d array')

        if len(points0.shape) != len(points1.shape):
            raise ValueError('points0 and points1 must have the same dimensions')

        if not hasattr(values, 'ndim'):
            raise ValueError('values is not a np array')

        if copyInput:
            values_view = values.view()
            values_view.shape = (values.size,)
            self.values = np.ctypeslib.as_array(Array(values.dtype.char,
                values_view, lock=False))
            self.values.shape = values.shape

            self.points0 = np.ctypeslib.as_array(Array(points0.dtype.char, points0,
                lock=False))
            self.points1 = np.ctypeslib.as_array(Array(points1.dtype.char, points1,
                lock=False))
        else:
            self.values = values
            self.points0 = points0
            self.points1 = points1

        # save the shape of values in a shared array
        values_shape = np.array(values.shape)
        self.values_shape = np.ctypeslib.as_array(Array(values_shape.dtype.char,
            values_shape, lock=False))


        # ensure everything is C contiguous, since we use a C extension
        # I don't use np.ascontiguous since when used with shared memory and
        # copyInput=False and not C-CONTIGUOUS array, np.ascontiguous would
        # copy the data
        if not self.values.flags['C_CONTIGUOUS']:
            raise ValueError('self.values is not C_CONTIGUOUS')

        if not self.points0.flags['C_CONTIGUOUS']:
            raise ValueError('self.point0 is not C_CONTIGUOUS')

        if not self.points1.flags['C_CONTIGUOUS']:
            raise ValueError('self.point1 is not C_CONTIGUOUS')

        # unnecessary
        assert self.values_shape.flags['C_CONTIGUOUS']


        d0 = self.points0.size
        d1 = self.points1.size

        if values.shape[0] != d0:
            raise ValueError

        if values.shape[1] != d1:
            raise ValueError

        # if len(points) > values.ndim:
            # raise ValueError("There are %d point arrays, but values has %d "
                             # "dimensions" % (len(points), values.ndim))

        # if hasattr(values, 'dtype') and hasattr(values, 'astype'):
            # if not np.issubdtype(values.dtype, np.inexact):
                # values = values.astype(float)

        if fill_value is None:
            self.fill_value = fill_value
        else:

            fill_value_dtype = np.asarray(fill_value).dtype
            if (hasattr(values, 'dtype')
                    and not np.can_cast(fill_value_dtype, values.dtype,
                                        casting='same_kind')):
                raise ValueError("fill_value must be either 'None' or "
                                 "of a type compatible with values")
            filltmp = np.array([fill_value])
            self.fill_value = np.ctypeslib.as_array(Array(filltmp.dtype.char,
                filltmp, lock=False))

        if not np.all(np.diff(self.points0) > 0.):
                raise ValueError("points0 must be strictly "
                                 "ascending")
        if not np.all(np.diff(self.points1) > 0.):
                raise ValueError("points1 must be strictly "
                                 "ascending")


        # self.d1 = self.values.shape[0]
        # self.d2 = self.values.shape[1]
        # self.d3 = self.values.shape[2]

        # in this way I get an error if the new shape involves a copy of the array
        self.values.shape = (values.size,)


    def __call__(self, xi0, xi1, method=None, pyth=True):
        """
        Interpolation at coordinates
        """

        method = self.method if method is None else method
        if method not in ["linear"]:
            raise ValueError("Method '%s' is not defined" % method)

        if len(xi0) != len(xi1):
            raise ValueError('xi0 must have the same size of xi1')

        # t1 = time.time()
        # do the search on the two arrays
        ind0 = np.searchsorted(self.points0, xi0)
        ind1 = np.searchsorted(self.points1, xi1)
        # print (time.time() - t1 , 's search')


        # t1 = time.time()
        if self.bounds_error:
            if np.any(xi0[ind0 == 0] < self.points0.min()):
                raise ValueError("One of the requested xi0 is out of bounds")
            if np.any(xi1[ind1 == 0] < self.points1.min()):
                raise ValueError("One of the requested xi1 is out of bounds")
            if np.any(ind0 == len(self.points0)):
                raise ValueError("One of the requested xi0 is out of bounds")
            if np.any(ind1 == len(self.points1)):
                raise ValueError("One of the requested xi1 is out of bounds")
        else:
            outofbounds = np.full_like(ind1, False, dtype=bool)
            outofbounds[xi0[ind0 == 0] < self.points0.min()] = True
            outofbounds[xi1[ind1 == 0] < self.points1.min()] = True
            outofbounds[ind0 == len(xi0)] = True
            outofbounds[ind1 == len(xi1)] = True

            ind0[outofbounds] = 1
            ind1[outofbounds] = 1

        # print (time.time() - t1 , 's check')


        # t1 = time.time()
        t = (xi0 - self.points0[ind0-1]) / (self.points0[ind0] -
                self.points0[ind0-1])
        u = (xi1 - self.points1[ind1-1]) / (self.points1[ind1] -
                self.points1[ind1-1])


        # out = np.zeros((len(xi0),self.d3), dtype=self.values.dtype)
        # print (time.time() - t1 , 's generation t and u')

        # for i, (i0, i1) in enumerate(zip(ind0, ind1)):
            # out[i,:] = ( (1.-t[i])*(1.-u[i])* self.values[i0-1, i1-1, :] +
                    # t[i]*(1.-u[i])*self.values[i0,i1-1,:] + t[i]*u[i]*self.values[i0,
                        # i1,:] 
                    # + (1.-t[i])*u[i]*self.values[i0-1, i1,:]
                    # )

    
        # from loopCyth import loopCython
        # t1 = time.time()
        if pyth:
            out = loopPhyton(self.values, t, u, ind0, ind1)
        else:
            # assert self.values.shape[0] == t.size
            # assert self.values.shape[1] == u.size
            assert ind0.size == t.size
            assert ind1.size == u.size
            # out = loopCython(self.values, t, u, ind0, ind1)

            # import ctypes
            # libloop = ctypes.CDLL('./libloopc.so')
            # libloop.connect()

            # d1 = self.values.shape[0]
            # d2 = self.values.shape[1]
            # d3 = self.values.shape[2]
            # npoints = t.size
            # out = np.zeros((t.size,d3), dtype=values.dtype)
            # # out= libloop.loopC(d1,d2,d3,npoints, values, t, u, ind0, ind1)
            # out= libloop.loopC(d1,d2,d3,npoints, values, t, u, ind0, ind1)

            # # self.values = np.asfortranarray(self.values.T)
            # # d1 = self.values.shape[0]
            # # d2 = self.values.shape[1]
            # # d3 = self.values.shape[2]
            # # npoints = t.size
            # # # out = np.zeros((t.size,d3), dtype=self.values.dtype, order='F')
            # # print (self.values.shape)
            # # out = np.zeros((d1, t.size), dtype=self.values.dtype, order='F')
            # # from loopF import loopf
            # # from loopFT import loopft
            # # loopft(self.values, t, u, ind1, ind0, out)

            # out = np.asarray(out.T, order='C')

            # C version
            out = loopC(self.values, t, u, ind0, ind1)
        # print (time.time() - t1 , 's internals')


        if not self.bounds_error and self.fill_value is not None:
            out[outofbounds, :] = self.fill_value

        return out

def init(points0, points1, values, method="linear", bounds_error=True,
             fill_value=np.nan):
    """"
    Test, do not use.

    """

    if method not in ["linear"]:
        raise ValueError("Method '%s' is not defined" % method)
    method = method
    bounds_error = bounds_error

    if not hasattr(values, 'ndim'):
        # allow reasonable duck-typed values
        values = np.asarray(values)

    points0 = np.asarray(points0)
    points1 = np.asarray(points1)

    if len(points0.shape) != 1:
        raise ValueError('points0 must be 1d array')

    if len(points1.shape) != 1:
        raise ValueError('points1 must be 1d array')

    if len(points0.shape) != len(points1.shape):
        raise ValueError('points0 and points1 must have the same dimensions')

    d0 = points0.size
    d1 = points1.size

    if values.shape[0] != d0:
        raise ValueError

    if values.shape[1] != d1:
        raise ValueError

    del d0, d1

    # if len(points) > values.ndim:
        # raise ValueError("There are %d point arrays, but values has %d "
                         # "dimensions" % (len(points), values.ndim))

    # if hasattr(values, 'dtype') and hasattr(values, 'astype'):
        # if not np.issubdtype(values.dtype, np.inexact):
            # values = values.astype(float)

    if fill_value is not None:
        fill_value_dtype = np.asarray(fill_value).dtype
        if (hasattr(values, 'dtype')
                and not np.can_cast(fill_value_dtype, values.dtype,
                                    casting='same_kind')):
            raise ValueError("fill_value must be either 'None' or "
                             "of a type compatible with values")

    if not np.all(np.diff(self.points0) > 0.):
            raise ValueError("points0 must be strictly "
                             "ascending")
    if not np.all(np.diff(self.points1) > 0.):
            raise ValueError("points1 must be strictly "
                             "ascending")


    # ensure that values is C contiguous
    # values = np.ascontiguousarray(values)
    if not values.flags['C_CONTIGUOUS']:
        raise ValueError('values must be C contiguous')

    d1 = values.shape[0]
    d2 = values.shape[1]
    d3 = values.shape[2]

    # in this way I get an error if the new shape involves a copy of the array
    self.values.shape = (self.d1*self.d2*self.d3,)

    return (method, bounds_error,points0, points1, d1, d2, d3, values,
            fill_value)

@njit(parallel=False, nogil=True)
def interpolate_call_jit(obj_points0, obj_points1, obj_bounds_error,
        obj_fill_value, obj_values, obj_values_shape,
        xi0, xi1, pyth=False):
    """
    Interpolation at coordinates, jitted version. In case the input is a
    scalar, it must be taken care outside of the function.

    """

    # get them from the shared array
    d1 = obj_values_shape[0]
    d2 = obj_values_shape[1]
    d3 = obj_values_shape[2]


    if np.asarray(xi0).ndim == 0 or np.asarray(xi1).ndim == 0:
        raise ValueError('For scalars, wrap them into arrays:'
        ' np.array([value])')

    if obj_values.ndim != 1:
        raise ValueError('obj_values must be the array directly from the object')

    xi0 = np.asarray(xi0)
    xi1 = np.asarray(xi1)

    if len(xi0) != len(xi1):
        raise ValueError('xi0 must have the same size of xi1')

    # t1 = time.time()
    # do the search on the two arrays, and output must be C contiguous, even if
    # we're calling the fortran part, where the it loops as in the C version
    ind0 = np.ascontiguousarray(np.searchsorted(obj_points0, xi0))
    ind1 = np.ascontiguousarray(np.searchsorted(obj_points1, xi1))
    # print (time.time() - t1 , 's search')


    # t1 = time.time()
    if obj_bounds_error[0]:
        if np.any(xi0[ind0 == 0] < obj_points0.min()):
            raise ValueError("One of the requested xi0 is out of bounds")
        if np.any(xi1[ind1 == 0] < obj_points1.min()):
            raise ValueError("One of the requested xi1 is out of bounds")
        if np.any(ind0 == len(obj_points0)):
            raise ValueError("One of the requested xi0 is out of bounds")
        if np.any(ind1 == len(obj_points1)):
            raise ValueError("One of the requested xi1 is out of bounds")
    else:
        outofbounds = np.full_like(ind1, False, dtype=np.bool_)
        outofbounds[xi0[ind0 == 0] < obj_points0.min()] = True
        outofbounds[xi1[ind1 == 0] < obj_points1.min()] = True
        outofbounds[ind0 == len(xi0)] = True
        outofbounds[ind1 == len(xi1)] = True

        ind0[outofbounds] = 1
        ind1[outofbounds] = 1

    ind00 = ind0 == 0
    ind10 = ind1 == 0

    corner00 = np.logical_and(ind00, ind10)
    cornerMax0 = np.logical_and(ind0==len(obj_points0), ind10)
    corner0Max = np.logical_and(ind00, ind1==len(obj_points1))
    cornerMaxMax = np.logical_and(ind0==len(obj_points0), ind1==len(obj_points1))
    firstindex = np.flatnonzero(np.logical_and(ind00, np.logical_not(ind10)))
    secondindex = np.flatnonzero(np.logical_and(np.logical_not(ind00), ind10))

    ind0_original = np.copy(ind0)
    ind1_original = np.copy(ind1)

    # print (time.time() - t1 , 's check')

    ind0[ind00] = 1
    ind1[ind10] = 1

    # t1 = time.time()
    t = np.ascontiguousarray((xi0 - obj_points0[ind0-1]) / (obj_points0[ind0] -
            obj_points0[ind0-1]))
    u = np.ascontiguousarray((xi1 - obj_points1[ind1-1]) / (obj_points1[ind1] -
            obj_points1[ind1-1]))

    ind0[np.logical_or(ind00, ind10)] = 1
    ind1[np.logical_or(ind00, ind10)] = 1

    if pyth:
        obj_values = obj_values.reshape((d1,d2,d3))
        out = loopPhytonNumba(obj_values, t, u, ind0, ind1, d1, d2, d3)
        obj_values = obj_values.reshape((d1*d2*d3,))
        out = loopPhytonFirstIndexNumba(obj_values, u, ind1_original, firstindex, out)
        out = loopPhytonSecondIndexNumba(obj_values, t, ind0_original, d2, secondindex, out)
    else:
        if ind0.size != t.size or ind1.size != u.size:
            raise ValueError

        # C version
        out = loopC2_jit(obj_values, t, u, ind0, ind1, d1, d2, d3,
                ind0_original, ind1_original, firstindex, secondindex)

    if not obj_bounds_error[0] and obj_fill_value is not None:
        out[outofbounds, :] = obj_fill_value


    values_arr = obj_values.reshape(d1,d2,d3)
    out[corner00] = values_arr[0,0,:]
    out[cornerMax0] = values_arr[d1-1,0,:]
    out[corner0Max] = values_arr[0,d2-1,:]
    out[cornerMaxMax] = values_arr[d1-1,d2-1,:]

    return out


def interpolate_call(obj, xi0, xi1, pyth=False, squeezeScalarInput
        = False, fortran=False):
    """
    Interpolation at coordinates

    """

    # get them from the shared array
    d1 = obj.values_shape[0]
    d2 = obj.values_shape[1]
    d3 = obj.values_shape[2]


    # method = obj.method if method is None else method
    # if method not in ["linear"]:
        # raise ValueError("Method '%s' is not defined" % method)

    # don't blindly use np.asarray() since I might pass a long list, and I
    # don't want to copy it.
    if np.ndim(xi0) == 0:
        xi0 = np.asarray([xi0])
    if np.ndim(xi1) == 0:
        xi1 = np.asarray([xi1])

    if not hasattr(xi0, 'ndim'):
        raise ValueError('xi0 is not a np array')

    if not hasattr(xi1, 'ndim'):
        raise ValueError('xi1 is not a np array')

    if len(xi0) != len(xi1):
        raise ValueError('xi0 must have the same size of xi1')

    # t1 = time.time()
    # do the search on the two arrays, and output must be C contiguous, even if
    # we're calling the fortran part, where the it loops as in the C version
    ind0 = np.ascontiguousarray(np.searchsorted(obj.points0, xi0))
    ind1 = np.ascontiguousarray(np.searchsorted(obj.points1, xi1))
    # print (time.time() - t1 , 's search')


    # t1 = time.time()
    if obj.bounds_error[0]:
        if np.any(xi0[ind0 == 0] < obj.points0.min()):
            raise ValueError("One of the requested xi0 is out of bounds")
        if np.any(xi1[ind1 == 0] < obj.points1.min()):
            raise ValueError("One of the requested xi1 is out of bounds")
        if np.any(ind0 == len(obj.points0)):
            raise ValueError("One of the requested xi0 is out of bounds")
        if np.any(ind1 == len(obj.points1)):
            raise ValueError("One of the requested xi1 is out of bounds")
    else:
        outofbounds = np.full_like(ind1, False, dtype=bool)
        outofbounds[xi0[ind0 == 0] < obj.points0.min()] = True
        outofbounds[xi1[ind1 == 0] < obj.points1.min()] = True
        outofbounds[ind0 == len(xi0)] = True
        outofbounds[ind1 == len(xi1)] = True

        ind0[outofbounds] = 1
        ind1[outofbounds] = 1

    ind00 = ind0 == 0
    ind10 = ind1 == 0

    corner00 = np.logical_and(ind00, ind10)
    cornerMax0 = np.logical_and(ind0==len(obj.points0), ind10)
    corner0Max = np.logical_and(ind00, ind1==len(obj.points1))
    cornerMaxMax = np.logical_and(ind0==len(obj.points0), ind1==len(obj.points1))
    firstindex = np.flatnonzero(np.logical_and(ind00, np.logical_not(ind10)))
    secondindex = np.flatnonzero(np.logical_and(np.logical_not(ind00), ind10))

    ind0_original = np.copy(ind0)
    ind1_original = np.copy(ind1)

    # print (time.time() - t1 , 's check')

    ind0[ind00] = 1
    ind1[ind10] = 1

    # t1 = time.time()
    t = np.ascontiguousarray((xi0 - obj.points0[ind0-1]) / (obj.points0[ind0] -
            obj.points0[ind0-1]))
    u = np.ascontiguousarray((xi1 - obj.points1[ind1-1]) / (obj.points1[ind1] -
            obj.points1[ind1-1]))

    ind0[np.logical_or(ind00, ind10)] = 1
    ind1[np.logical_or(ind00, ind10)] = 1

    if pyth:
        
        values_arr = obj.values.reshape((d1,d2,d3))
        out = loopPhytonNumba(values_arr, t, u, ind0, ind1, d1, d2, d3)
        values_arr.shape = (d1*d2*d3)
        out = loopPhytonFirstIndexNumba(values_arr, u, ind1_original, firstindex, out)
        out = loopPhytonSecondIndexNumba(values_arr, t, ind0_original, d2, secondindex, out)
    else:
        # assert obj.values.shape[0] == t.size
        # assert obj.values.shape[1] == u.size
        assert ind0.size == t.size
        assert ind1.size == u.size

        # # C version
        # out = loopC(obj.values, t, u, ind0, ind1)

        if fortran:
            # fortran version
            out = loopF(obj.values, t, u, ind0, ind1, d1, d2, d3,
                    ind0_original, ind1_original, firstindex, secondindex)
        else:
            # C version
            out = loopC2(obj.values, t, u, ind0, ind1, d1, d2, d3,
                    ind0_original, ind1_original, firstindex, secondindex)

        # python part for debug, I should move it in the python loop ------
        # values_arr = obj.values.reshape(d1,d2,d3)

        # for fi in firstindex:
            # out[fi,:] = ((1.-u[fi])* values_arr[0,ind1_original[fi]-1,:] +
                    # u[fi]*values_arr[0,ind1_original[fi],:])

        # for si in secondindex:
            # out[si,:] = ((1.-t[si])* values_arr[ind0_original[si]-1,0,:] +
                    # t[si]*values_arr[ind0_original[si],0,:])
        # ----------------------------------------------------------------
        
    # print (time.time() - t1 , 's internals')


    if not obj.bounds_error[0] and obj.fill_value is not None:
        out[outofbounds, :] = obj.fill_value


    values_arr = obj.values.reshape(d1,d2,d3)
    out[corner00] = values_arr[0,0,:]
    out[cornerMax0] = values_arr[d1-1,0,:]
    out[corner0Max] = values_arr[0,d2-1,:]
    out[cornerMaxMax] = values_arr[d1-1,d2-1,:]

    if xi1.size == 1 and squeezeScalarInput:
        # scalar input
        out = np.squeeze(out)

    return out


def loopPhyton(values, t, u, ind0, ind1, d1,d2,d3):
    values.shape = (d1,d2,d3)
    out = np.zeros((t.size,d3), dtype=values.dtype)
    out[:,:] = [ ( (1.-t[i])*(1.-u[i])* values[i0-1, i1-1, :] +
            t[i]*(1.-u[i])*values[i0,i1-1,:] + t[i]*u[i]*values[i0,
                i1,:] 
            + (1.-t[i])*u[i]*values[i0-1, i1,:]
            ) for i, (i0, i1) in enumerate(zip(ind0, ind1))]

    values.shape = (d1*d2*d3)

    return out

@njit(parallel=False, nogil=True)
def loopPhytonNumba(values, t, u, ind0, ind1, d1,d2,d3):
    out = np.zeros((t.size,d3), dtype=values.dtype)

    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i,j] = ( (1.-t[i])*(1.-u[i])* values[ind0[i]-1, ind1[i]-1, j] +
                    t[i]*(1.-u[i])*values[ind0[i],ind1[i]-1,j] + t[i]*u[i]*values[ind0[i],
                        ind1[i],j] + (1.-t[i])*u[i]*values[ind0[i]-1, ind1[i],j]) 


    return out

@njit(parallel=False, nogil=True)
def loopPhytonFirstIndexNumba(values, u, ind1, firstIndex, out):

    d3 = out.shape[1]
    for i in range(firstIndex.size):
        for j in range(out.shape[1]):
            ii = firstIndex[i]
            out[ii,j] = ((1.-u[ii]) * values[(ind1[ii]-1)*d3+j] +
                    u[ii] * values[ind1[ii]*d3+j])
    return out

@njit(parallel=False, nogil=True)
def loopPhytonSecondIndexNumba(values, t, ind0, d2, secondIndex, out):

    d3 = out.shape[1]
    for i in range(secondIndex.size):
        for j in range(out.shape[1]):
            ii = secondIndex[i]
            out[ii,j] = ((1.-t[ii]) * values[(ind0[ii]-1)*d2*d3+  j] +
                    t[ii] * values[ind0[ii]*d2*d3+j])

    return out


# def loopPhyton(values, t, u, ind0, ind1):
    # d1 = values.shape[0]
    # d2 = values.shape[1]
    # d3 = values.shape[2]
    # # a=np.copy(values, order='C')
    # # assert np.array_equal(a, values)
    # values.shape = (d1*d2*d3,)
    # out = np.zeros((t.size*d3), dtype=values.dtype)
    # one = 1.

    # # for i in xrange(t.size):
        # # for j in xrange(d3):
            # # out[i*d3+j] = ((one-t[i]) * (one-u[i]) *                     
                    # # values[(ind0[i]-1)*d2*d3+ (ind1[i]-1)*d3+ j] +               
                    # # t[i] * (one-u[i]) * values[ind0[i]*d2*d3+(ind1[i]-1)*d3+j]
                    # # + t[i] * u[i] * values[ind0[i]*d2*d3 + ind1[i]*d3 + j] +
                    # # (one-t[i]) * u[i] * values[(ind0[i]-1)*d2*d3+ ind1[i]*d3+j])
    # # for i in xrange(d1):
        # # for j in xrange(d2):
            # # for k in xrange(d3):
                # # print (a[i,j,k], values[i*d2*d3+j*d3+k])
    # print ('python ind0')
    # for i in xrange(t.size):
        # for j in xrange(d3):
            # print(values[(ind0[i]-1)*d2*d3+ (ind1[i]-1)*d3+ j])
        

    # out = np.ascontiguousarray(out)

    # out.shape = (t.size, d3)
    # values.shape = (d1,d2,d3)
    # return out

def loopF(values, t, u, ind0, ind1, d1, d2, d3, 
        ind0_original, ind1_original, firstindex, secondindex):

    out = np.zeros((t.size*d3,), dtype=values.dtype)
    assert np.can_cast(ind0.max(),np.int32)
    assert np.can_cast(ind1.max(),np.int32)

    assert np.can_cast(ind0.min(),np.int32)
    assert np.can_cast(ind1.min(),np.int32)

    # ind0 = ind0.astype(np.int32)
    # ind1 = ind1.astype(np.int32)

    from .loopf import loopf
    from .loopffirstIndex import loopffirstindex
    from .loopfsecondIndex import loopfsecondindex
    m = t.size*d3
    loopf(d1,d2,d3,values,t,u,ind0,ind1, out)
    loopffirstindex (d1,d2,d3,values,u,firstindex, ind1_original, out)
    loopfsecondindex(d1,d2,d3,values,t,ind0_original,secondindex, out)
    out = np.ascontiguousarray(out)

    out.shape = (t.size, d3)
    # values.shape = (d1,d2,d3)

    return out

@njit(parallel=False, nogil=True)
def loopC2_jit(values, t, u, ind0, ind1, d1, d2, d3, 
        ind0_original, ind1_original, firstindex, secondindex):

    out = np.zeros((t.size*d3,), dtype=values.dtype)

    if np.any(ind0<0):
        raise ValueError('for some reason ind0 contains <0')
    if np.any(ind1<0):
        raise ValueError('for some reason ind1 contains <0')

    m = t.size*d3
    n = values.size
    npoints = t.size
    loop_ffi(d1,d2,d3,npoints,ffi.from_buffer(values),ffi.from_buffer(t),
            ffi.from_buffer(u),ffi.from_buffer(ind0), 
            ffi.from_buffer(ind1), ffi.from_buffer(out))
    loopfirstindex_ffi (n, npoints, out.size, firstindex.size, 
            d1,d2,d3,ffi.from_buffer(values),ffi.from_buffer(u),
            ffi.from_buffer(firstindex), ffi.from_buffer(ind1_original),
            ffi.from_buffer(out))
    loopsecondindex_ffi(n, npoints, out.size, secondindex.size,
            d1,d2,d3, ffi.from_buffer(values),ffi.from_buffer(t),
            ffi.from_buffer(ind0_original), ffi.from_buffer(secondindex),
            ffi.from_buffer(out))

    out = np.ascontiguousarray(out)

    # usually I would do modify the out.shape, to ensure that we are not
    # actually copying the array, but since it's contiguous, it should not copy
    # it (changing the array shape does not work with numba)
    # out.shape = (t.size, d3)
    out = out.reshape((t.size, d3))

    return out

def loopC2(values, t, u, ind0, ind1, d1, d2, d3, 
        ind0_original, ind1_original, firstindex, secondindex):

    out = np.zeros((t.size*d3,), dtype=values.dtype)
    assert np.can_cast(ind0.max(),np.int32)
    assert np.can_cast(ind1.max(),np.int32)

    assert np.can_cast(ind0.min(),np.int32)
    assert np.can_cast(ind1.min(),np.int32)

    import ctypes
    from numpy.ctypeslib import ndpointer

    # from pathlib import PurePath
    # path_lib = PurePath('../egl/interp')

    from .get_project_root import get_project_root
    path_lib = get_project_root()

    lib = ctypes.cdll.LoadLibrary(path_lib / "loopC.so")
    loop=lib.loop
    loopfirstindex = lib.loopfirstIndex
    loopsecondindex = lib.loopsecondIndex

    loop.restype = None
    loopfirstindex.restype = None
    loopsecondindex.restype = None

    loop.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_ssize_t, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_ssize_t, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

    loopfirstindex.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, 
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_ssize_t, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_ssize_t, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

    loopsecondindex.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, 
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_ssize_t, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_ssize_t, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

    m = t.size*d3
    n = values.size
    npoints = t.size
    loop(d1,d2,d3,npoints,values,t,u,ind0,ind1, out)
    loopfirstindex (n, npoints, out.size, firstindex.size, 
            d1,d2,d3,values,u,firstindex, ind1_original, out)
    loopsecondindex(n, npoints, out.size, secondindex.size,
            d1,d2,d3,values,t,ind0_original,secondindex, out)

    out = np.ascontiguousarray(out)

    out.shape = (t.size, d3)
    # values.shape = (d1,d2,d3)

    return out




def loopC(values, t, u, ind0, ind1):
    # ensure that values is C contiguous
    d1 = values.shape[0]
    d2 = values.shape[1]
    d3 = values.shape[2]

    values = values.reshape((d1*d2*d3))
    out = np.zeros((t.size*d3,), dtype=values.dtype)
    assert np.can_cast(ind0.max(),np.int32)
    assert np.can_cast(ind1.max(),np.int32)

    from loopc import loopc
    out = loopc(out.size, values, t, u, ind0, ind1, d1,d2,d3)

    # in this way I get an error if the new shape involves a copy of the array
    out.shape = (t.size, d3)

    return out

if __name__ ==  '__main__':

    raiseInterpError = True

    # test the values
    x0 = np.linspace(0., 10., 6)
    x1 = np.logspace(2., 5., 8)
    sed = np.random.rand(x0.size, x1.size, 50000)

    from scipy.interpolate import RegularGridInterpolator
    interpFunctionYoung = RegularGridInterpolator(
            (x0, x1,), sed,
            method='linear', bounds_error=raiseInterpError, fill_value=None)




    N = 10
    xi0 = np.linspace(0., 10., N)
    xi1 = np.logspace(2., 5., N)
    xi0 = np.concatenate([xi0, np.zeros((N,))])
    xi1 = np.concatenate([xi1, np.logspace(2., 5., N)])

    xi0 = np.concatenate([xi0, np.linspace(0., 10., N)])
    xi1 = np.concatenate([xi1, np.full((N,),100.)])

    pts = np.array([[xi0[i], xi1[i]] for i in range(xi0.size)])
    
    t = time.time()
    sed_interp = interpFunctionYoung(pts)
    print ('regular grid interpolator shape ',sed_interp.shape)
    print (time.time() - t, 's interpolation scipy')

    myinterp = interpolate2_3D(x0, x1, sed, bounds_error=raiseInterpError)


    # t = time.time()
    # sed_interpmy = myinterp(xi0, xi1, pyth=True)
    # print (sed_interpmy.shape)
    # print (time.time() - t, 's pure python+numpy')

    # if (np.all(np.isclose(sed_interp, sed_interpmy))):
        # print ('scipy and pure python+numpy same result')
    # else:
        # print ('scipy and pure python+numpy DIFFERENT result')

    # t = time.time()
    # sed_interpmy = interpolate_call(myinterp,xi0, xi1, pyth=True)
    # print ('correct output shape', sed_interpmy.shape)
    # print (time.time() - t, 's pure python+numpy separate call')

    # if (np.all(np.isclose(sed_interp, sed_interpmy))):
        # print ('scipy and pure python+numpy separate call same result')
    # else:
        # print ('scipy and pure python+numpy separate call DIFFERENT result')



    t = time.time()
    # sed_interpmy = myinterp(xi0, xi1, pyth=False)
    sed_interpmy = interpolate_call(myinterp,xi0, xi1, pyth=False)
    print (sed_interpmy.shape)
    print (time.time() - t, 's fortran')


    if (np.all(np.isclose(sed_interp, sed_interpmy))):
        print ('scipy and fortran same result')
    else:
        print ('scipy and fortran DIFFERENT result')

    # for i in xrange(xi1.size):
        # for j in xrange(sed_interp.shape[1]):
            # print (sed_interp[i,j], sed_interpmy[i,j], np.isclose(sed_interp[i,j],
                # sed_interpmy[i,j]))
    print ('test on shape when the input is a scalar')
    sed_interpmy = interpolate_call(myinterp,5., 200., pyth=False,
            squeezeScalarInput=True)

    if sed_interpmy.shape == (myinterp.d3,):
        print ('scalar test ok')
    print (np.squeeze(sed_interpmy.shape))
    assert sed_interpmy.shape == (myinterp.d3,)
