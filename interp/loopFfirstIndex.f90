      subroutine loopFfirstIndex(n, npoints, m, l, d1, d2, d3,          &
     & values, u, firstIndex, ind1, output)
      !use iso_c_binding, only: c_long, c_double
      implicit none
      integer, parameter :: rl=kind(1.d0)
      integer, intent(in) :: d1, d2, d3, npoints, m, n, l

      ! zero-based like in C
      real(rl), intent(in) :: values(0:n-1)
      real(rl), intent(in) :: u(0:npoints-1)
      integer*8, intent(in) :: ind1(0:npoints-1)
      integer*8, intent(in) :: firstIndex(0:l-1)
      real(rl), intent(inout) :: output(0:m-1)
      real(rl), parameter :: one=1._rl

      integer :: i,j, ii

      !must be n = d1*d2*d3
      !must be m = npoints * d3
      !values[d1][d2][d3] and out[npoints][d3]

      
      do i=0, l-1
        do j=0, d3-1
          ii = firstIndex(i)
          output(ii*d3+j) =             (one-u(ii)) *                   &
     &      values( (ind1(ii)-1)*d3+ j) +                               &
     &      u(ii) * values(ind1(ii)*d3+j)
        enddo
      enddo

      endsubroutine
