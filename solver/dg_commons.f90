module dg_commons
  ! DG solver parameters
  !integer,parameter::n=3
  integer::n,nx,nquad,ninit,bc
  character(LEN=3),parameter::integrator='RK2'
  !integer,parameter::nx=64
  !integer,parameter::nquad=n
  integer,parameter::nvar=1
  integer,parameter::riemann=1
  logical,parameter::use_limiter=.false.
  ! Problem set-up
  !integer,parameter::ninit=2
  !integer,parameter::bc=1
  real(kind=8)::tend=1.0
  real(kind=8)::boxlen=1.0
  real(kind=8)::gamma=5./3.
  real(kind=8)::freq_dt=0.05
  character(LEN=*),parameter::base_folder='outputs/'
  character(LEN=50)::title!="outputs/disc_64_rk2_dg3"

  ! Misc commons
  real(kind=8),dimension(:),allocatable::chsi_quad,w_quad

end module dg_commons
