#include <stddef.h>
#include <sys/types.h>
/*[>static void loopC(int d1, int d2, int d3, int npoints, double values[d1][d2][d3], double t[npoints],  double u[npoints], long ind0[npoints], long ind1[npoints], double out[npoints][d3]) {<]*/
/*static void loopC(int d1, int d2, int d3, int npoints, double values[d1][d2][d3], double t[npoints],  double u[npoints], long ind0[npoints], long ind1[npoints]) {*/

  /*long i, j;*/
  /*double out[npoints][ d3];*/

  /*for (i=0; i<npoints; i++) {*/
    /*for (j=0; j<d3; j++) {*/
      /*out[i][j] = (1.-t[i]) * (1.-u[i]) * values[ind0[i]-1][ ind1[i]-1][ j] + t[i] * (1.-u[i]) * values[ind0[i]][ind1[i]-1][j] + t[i] * u[i] * values[ind0[i]][ ind1[i]][j] + (1.-t[i]) * u[i] * values[ind0[i]-1][ ind1[i]][j];*/
    /*}*/
  /*}*/
  /*return out;*/
/*}*/

static long ind3d(long d1, long d2, long i, long j, long k)
{
  long index;
  index = i*d1*d2 + j*d2 + k;
  return (index);
}


void loop(const size_t d0, const size_t d1, const size_t d2, const size_t npoints, 
    const double values[], const double t[],  const double u[], const ssize_t ind0[], 
    const ssize_t ind1[], double out[])
{

  size_t i, j;

  for (i=0; i<npoints; i++) 
  {
    for (j=0; j<d2; j++) {
      out[i*d2+j] = (1.-t[i]) * (1.-u[i]) * 
        values[ind3d(d1,d2,ind0[i]-1, ind1[i]-1, j)] + 
        t[i] * (1.-u[i]) * values[ind3d(d1,d2,ind0[i],ind1[i]-1,j)] + 
        t[i] * u[i] * values[ind3d(d1,d2,ind0[i], ind1[i],j)] + 
        (1.-t[i]) * u[i] * values[ind3d(d1,d2,ind0[i]-1, ind1[i],j)];
    }
  }
}



void loopfirstIndex(const size_t n, const size_t npoints, const size_t m, const size_t l, 
    const size_t d0, const size_t d1, const size_t d2,  const double values[], 
    const double u[], const ssize_t firstIndex[], const ssize_t ind1[], double output[])
{

  size_t i, j;

  for (i=0; i<l; i++)
  {
    for (j=0; j<d2; j++)
    {
      output[firstIndex[i]*d2+j] = (1.-u[firstIndex[i]]) *
        values[ (ind1[firstIndex[i]]-1)*d2+ j] +
        u[firstIndex[i]] * values[ind1[firstIndex[i]]*d2+j];
    }


  }

}

void loopsecondIndex(const size_t n, const size_t npoints, const size_t m, const size_t l, 
    const size_t d0, const size_t d1, const size_t d2,  const double values[], 
    const double t[], const ssize_t ind0[], const ssize_t secondIndex[], double output[])
{

  size_t i, j;

  for (i=0; i<l; i++)
  {
    for (j=0; j<d2; j++)
    {
      output[secondIndex[i]*d2+j] = (1.-t[secondIndex[i]]) *
        values[ (ind0[secondIndex[i]]-1)*d1*d2+ j] +
        t[secondIndex[i]] * values[ind0[secondIndex[i]]*d1*d2+j];
    }


  }

}
