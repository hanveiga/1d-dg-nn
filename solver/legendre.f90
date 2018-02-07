function legendre(x,n)
  integer::n
  real(kind=8)::x
  real(kind=8)::legendre
  x=min(max(x,-1.0),1.0)
  select case(n)
  case(0)
     legendre=1.0
  case(1)
     legendre=x
  case(2)
     legendre=0.5*(3.0*x**2-1.0)
  case(3)
     legendre=0.5*(5.0*x**3-3.0*x)
  case(4)
     legendre=0.125*(35.0*x**4-30.0*x**2+3.0)
  case(5)
     legendre=0.125*(63.0*x**5-70.0*x**3+15.0*x)
  case(6)
     legendre=1.0/16.0*(231.0*x**6-315.0*x**4+105.0*x**2-5.0)
  end select
  legendre=sqrt(2.0*dble(n)+1.0)*legendre
  return
end function legendre

function legendre_prime(x,n)
  integer::n
  real(kind=8)::x
  real(kind=8)::legendre_prime
  x=min(max(x,-1.0),1.0)
  select case(n)
  case(0)
     legendre_prime=0.0
  case(1)
     legendre_prime=1.0
  case(2)
     legendre_prime=3.0*x
  case(3)
     legendre_prime=0.5*(15.0*x**2-3.0)
  case(4)
     legendre_prime=0.125*(140.0*x**3-60.0*x)
  case(5)
     legendre_prime=0.125*(315.0*x**4-210.0*x**2+15.0)
  case(6)
     legendre_prime=1.0/16.0*(1386.0*x**5-1260.0*x**3+210.0*x)
  end select
  legendre_prime=sqrt(2.0*dble(n)+1.0)*legendre_prime
  return
end function legendre_prime

subroutine gl_quadrature(x_quad,w_quad,n)
  integer::n
  real(kind=8),dimension(1:n)::x_quad,w_quad

  integer::i,iter
  real(kind=8)::dpi=acos(-1.0d0),xx
  real(kind=8)::legendre,legendre_prime

  write(*,*)"Computing Gauss-Legendre quadrature points and weights."
  do i=1,n
     xx=(1.0-0.125/n/n+0.125/n/n/n)* & 
          & cos(dpi*(4.0*dble(i)-1.0)/(4.0*dble(n)+2.0))
     do iter=1,50
        xx=xx-legendre(xx,n)/legendre_prime(xx,n)
     end do
     x_quad(i)=xx
     w_quad(i)=2.0*(2.0*dble(n)+1.0)/(1.0-x_quad(i)**2) &
          & /legendre_prime(x_quad(i),n)**2
  end do
  do i=n/2+1,n
     x_quad(i)=-x_quad(n-i+1)
     w_quad(i)=w_quad(n-i+1)
  end do
!!$  do i=1,n
!!$     write(*,*)i,x_quad(i),w_quad(i)
!!$  end do

end subroutine gl_quadrature
