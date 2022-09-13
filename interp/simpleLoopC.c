#include <stddef.h>
#include <sys/types.h>

void loopTestC(const size_t n, const double values[], double output[])
{

  size_t i;

  for (i=0; i<n; i++)
  {
      output[i] = values[i];
  }

}

void loopTestC2(const size_t n, double output[])
{

  size_t i;

  for (i=0; i<n; i++)
  {
      output[i] = output[i]+2;
  }

}
