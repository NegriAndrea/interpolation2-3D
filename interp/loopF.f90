       ! subroutine loopF(d1, d2, d3, npoints, values, t, u, ind0, ind1,   &
       !&   output)
       ! !use iso_c_binding, only: c_long, c_double
       ! implicit none
       ! integer, parameter :: rl=kind(1.d0)
       ! integer, intent(in) :: d1, d2, d3, npoints

       ! real(rl), intent(in) :: values(d1, d2, d3)
       ! real(rl), intent(in) :: t(npoints), u(npoints)
       ! integer, intent(in) :: ind0(npoints), ind1(npoints)
       ! real(rl), intent(inout) :: output(npoints,  d3)
       ! real(rl), parameter :: one=1._rl

       ! integer :: i,j

       ! 
       ! do j=1, d3
       !   do i=1, npoints
       !     output(i,j) = (one-t(i)) * (one-u(i)) *                       &
       !&      values(ind0(i)-1, ind1(i)-1, j) +                           &
       !&      t(i) * (one-u(i)) * values(ind0(i),ind1(i)-1,j) + t(i) *    &
       !&      u(i) * values(ind0(i), ind1(i),j) +                         &
       !&      (one-t(i)) * u(i) * values(ind0(i)-1, ind1(i),j)
       !   enddo
       ! enddo

       ! endsubroutine

      subroutine loopF(n, npoints, m, d1, d2, d3, values, t, u,         &
     &  ind0, ind1, output)
      !use iso_c_binding, only: c_long, c_double
      implicit none
      integer, parameter :: rl=kind(1.d0)
      integer, intent(in) :: d1, d2, d3, npoints, m, n

      ! zero-based like in C
      real(rl), intent(in) :: values(0:n-1)
      real(rl), intent(in) :: t(0:npoints-1), u(0:npoints-1)
      integer*8, intent(in) :: ind0(0:npoints-1), ind1(0:npoints-1)
      real(rl), intent(inout) :: output(0:m-1)
      real(rl), parameter :: one=1._rl

      integer :: i,j

      !must be n = d1*d2*d3
      !must be m = npoints * d3
      !values[d1][d2][d3] and out[npoints][d3]

      
      do i=0, npoints-1
        do j=0, d3-1
          output(i*d3+j) = (one-t(i)) * (one-u(i)) *                    &
     &      values((ind0(i)-1)*d2*d3+ (ind1(i)-1)*d3+ j) +              &
     &      t(i) * (one-u(i)) * values(ind0(i)*d2*d3+(ind1(i)-1)*d3+j)  &
     &      + t(i) * u(i) * values(ind0(i)*d2*d3 + ind1(i)*d3 + j) +    &
     &      (one-t(i)) * u(i) * values((ind0(i)-1)*d2*d3+ ind1(i)*d3+j)
        enddo
      enddo

      endsubroutine

