#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import ctypes
from numba import jit, njit
from multiprocessing import Array, Pool
from functools import partial

lib = ctypes.cdll.LoadLibrary('./simpleLoopC.so')
loop=lib.loopTestC

loop.restype = None

loop.argtypes = [ctypes.c_size_t,
        np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]


# loop.argtypes = [ctypes.c_size_t,ctypes.POINTER(ctypes.c_double),
        # ctypes.POINTER(ctypes.c_double)]

# loop.argtypes = [types.long, types.CPointer(types.double),
        # types.CPointer(types.double)]

#----------------------------
from cffi import FFI
ffi = FFI()
ffi.cdef('void loopTestC(size_t n, double values[], double output[]);')
ffi.cdef('void loopTestC2(size_t n, double* output);')
C = ffi.dlopen('./simpleLoopC.so')
loopTestC = C.loopTestC
loopTestC2 = C.loopTestC2

@njit(parallel=False)
def testC():
    a = np.arange(100, dtype=np.float64)
    b = np.zeros_like(a)

    print(a, b)
    n = a.size
    # loop(n, a, b)

    # pta = ffi.cast('double *', FFI.from_buffer(a))
    # ptb = ffi.cast('double *', FFI.from_buffer(b))

    pta = ffi.from_buffer(a)
    ptb = ffi.from_buffer(b)
    # pta = ffi.from_buffer(a.astype(np.float64))
    # ptb = ffi.from_buffer(b.astype(np.float64))

    # pta = ffi.cast('double *', a.ctypes.data)
    # ptb = ffi.cast('double *', b.ctypes.data)

    # pta = ffi.cast('double *', ffi.from_buffer(a))
    # ptb = ffi.cast('double *', ffi.from_buffer(b))

    loopTestC(n, pta, ptb)
    # print(ctypes.c_void_p(a.ctypes.data))
    # loop(ctypes.c_int64(n), ctypes.c_void_p(a.ctypes.data), ctypes.c_void_p(b.ctypes.data))
    # print(ctypes.c_int64)
    print(a, b)

# @njit(parallel=False)
def interface(a,b):
    n = a.size
    pta = ffi.from_buffer(a)
    ptb = ffi.from_buffer(b)
    loopTestC(n, pta, ptb)

@njit(parallel=False)
def interface2(ini, end, a,b):
    a2 = a[ini:end]
    b2 = b[ini:end]

    n = a2.size
    pta = ffi.from_buffer(a2)
    ptb = ffi.from_buffer(b2)
    loopTestC(n, pta, ptb)

    loopTestC2(n, pta)


# @njit(parallel=False)
def workermy(i):
    global off, dim, a, b
    interface2(off[i], off[i]+dim[i], a, b)

# @njit(parallel=False)
def testC_parallel():
    from . import chopArray
    global off, dim, a, b
    n = 200
    a = np.ctypeslib.as_array(Array('d',np.arange(n, dtype=np.float64), lock=False))
    b = np.ctypeslib.as_array(Array('d',n , lock=False))

    # a = 
    # b = np.zeros_like(a)
    print(a, b)


    off, dim = chopArray(a.size, 4)

    N=2
    if N==1:
        for i in range(4):
            workermy(i)
    else:
        pool = Pool(N)
        pool.imap(workermy, list(range(4)))
        pool.close()
        pool.join()
    print(a, b)


# testC_parallel()

a = np.arange(100, dtype=np.float64)
b = np.zeros_like(a)
interface(a,b)
a.reshape((50,2))
print(a.shape)
print(a,b)


# testC()
