#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import time

from egl.interp.interpolate2d import (interpolate2_3D, interpolate_call,
        interpolate_call_jit)

raiseInterpError = True

# test the values
x0 = np.linspace(0., 10., 6)
x1 = np.logspace(2., 5., 8)
sed = np.random.rand(x0.size, x1.size, 50000)

from scipy.interpolate import RegularGridInterpolator
interpFunctionYoung = RegularGridInterpolator(
        (x0, x1,), sed,
        method='linear', bounds_error=raiseInterpError, fill_value=None)




N = 3000
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

# --------
# this is a call just to initialize numba
xi0tmp = np.linspace(0., 10., 5)
xi1tmp = np.logspace(2., 5., 5)
sed_interpmy = interpolate_call(myinterp,xi0tmp, xi1tmp, pyth=True)
# --------

t = time.time()
sed_interpmy = interpolate_call(myinterp,xi0, xi1, pyth=True)
print (time.time() - t, 's pure python+numpy separate call')

assert sed_interp.shape == sed_interpmy.shape
if (np.all(np.isclose(sed_interp, sed_interpmy))):
    print ('scipy and pure python+numpy separate call same result')
else:
    print ('scipy and pure python+numpy separate call DIFFERENT result')

# import h5py
# with h5py.File('test_result_python', 'w') as ff:
    # ff['scipy'] = sed_interp
    # ff['numpy'] = sed_interpmy




t = time.time()
# sed_interpmy = myinterp(xi0, xi1, pyth=False)
sed_interpmy = interpolate_call(myinterp,xi0, xi1, pyth=False, fortran=True)
print (sed_interpmy.shape)
print (time.time() - t, 's fortran')


assert sed_interp.shape == sed_interpmy.shape
if (np.all(np.isclose(sed_interp, sed_interpmy))):
    print ('scipy and fortran same result')
else:
    print ('scipy and fortran DIFFERENT result')


t = time.time()
# sed_interpmy = myinterp(xi0, xi1, pyth=False)
sed_interpmyC = interpolate_call(myinterp,xi0, xi1, pyth=False, fortran=False)
print (sed_interpmyC.shape)
print (time.time() - t, 's C')

assert sed_interp.shape == sed_interpmyC.shape
if (np.all(np.isclose(sed_interp, sed_interpmyC))):
    print ('scipy and C same result')
else:
    print ('scipy and C DIFFERENT result')
del sed_interpmyC

# --------
# this is a call just to initialize numba
xi0tmp = np.linspace(0., 10., 5)
xi1tmp = np.logspace(2., 5., 5)
sed_interpmy = interpolate_call_jit(myinterp.points0, myinterp.points1,
        myinterp.bounds_error, myinterp.fill_value, myinterp.values,
        myinterp.values_shape,
        xi0tmp, xi1tmp, pyth=False)
sed_interpmy = interpolate_call_jit(myinterp.points0, myinterp.points1,
        myinterp.bounds_error, myinterp.fill_value, myinterp.values,
        myinterp.values_shape,
        xi0tmp, xi1tmp, pyth=True)
# --------

t = time.time()
sed_interpmyCjit = interpolate_call_jit(myinterp.points0, myinterp.points1,
        myinterp.bounds_error, myinterp.fill_value, myinterp.values,
        myinterp.values_shape,
        xi0, xi1, pyth=False)
print (sed_interpmyCjit.shape)
print (time.time() - t, 's C jitted call')

assert sed_interp.shape == sed_interpmyCjit.shape
if (np.all(np.isclose(sed_interp, sed_interpmyCjit))):
    print ('scipy and C jit call same result')
else:
    print ('scipy and C jit call DIFFERENT result')

t = time.time()
sed_interpmyjit = interpolate_call_jit(myinterp.points0, myinterp.points1,
        myinterp.bounds_error, myinterp.fill_value, myinterp.values,
        myinterp.values_shape,
        xi0, xi1, pyth=True)
print (sed_interpmyjit.shape)
print (time.time() - t, 's python jitted with jitted call')

assert sed_interp.shape == sed_interpmyCjit.shape
if (np.all(np.isclose(sed_interp, sed_interpmyCjit))):
    print ('scipy and python jitted with jitted call same result')
else:
    print ('scipy and python jitted with jitted call DIFFERENT result')



# for i in xrange(xi1.size):
    # for j in xrange(sed_interp.shape[1]):
        # print (sed_interp[i,j], sed_interpmy[i,j], np.isclose(sed_interp[i,j],
            # sed_interpmy[i,j]))
print ('test on shape when the input is a scalar')
sed_interpmy = interpolate_call_jit(myinterp.points0, myinterp.points1,
        myinterp.bounds_error, myinterp.fill_value, myinterp.values,
        myinterp.values_shape,
        np.array([5.]), np.array([200.]), pyth=False)

if sed_interpmy.shape[0] == 1:
    sed_interpmy = sed_interpmy[0,:]

if sed_interpmy.shape == (myinterp.values_shape[2],):
    print ('scalar test ok')
assert sed_interpmy.shape == (myinterp.values_shape[2],)

print ('test on shape when the input is a scalar')
sed_interpmy = interpolate_call(myinterp,5., 200., pyth=False,
        squeezeScalarInput=True)

if sed_interpmy.shape == (myinterp.values_shape[2],):
    print ('scalar test ok')
print (np.squeeze(sed_interpmy.shape))
assert sed_interpmy.shape == (myinterp.values_shape[2],)
