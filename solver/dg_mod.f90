program dg
  use dg_commons
  implicit none
  !==============================================
  ! This program solves the 1D Euler equations
  ! using the Discontinuous Galerkin method
  ! up to 4th order
  ! (using Legendre polynomials up to x^3).
  ! Romain Teyssier, November 1st 2015
  !==============================================

  real(kind=8),dimension(:,:,:),allocatable::u,dudt,w1,w2,w3,w4,w5,w6, ureal
  real(kind=8),dimension(:,:,:),allocatable::f1,f2,f3,f4,f5,f6,f7

  integer::iter=0,icell,i,j,ivar, counting
  real(kind=8)::legendre,legendre_prime
  real(kind=8)::xcell,dx,x_quad
  real(kind=8)::t,dt,cmax, accum_time
  real(kind=8)::start,end
  real(kind=8),dimension(1:nvar)::uu,ww
  character(len=100)::filename

  call read_params()
  call load_nn()

  allocate(u(nvar,n,nx),dudt(nvar,n,nx),w1(nvar,n,nx),w2(nvar,n,nx),w3(nvar,n,nx),&
          &w4(nvar,n,nx),w5(nvar,n,nx),w6(nvar,n,nx),ureal(nvar,n,nx))
  allocate(f1(nvar,n,nx),f2(nvar,n,nx),f3(nvar,n,nx),f4(nvar,n,nx),f5(nvar,n,nx),&
          &f6(nvar,n,nx),f7(nvar,n,nx))
  allocate(chsi_quad(n),w_quad(n))
  allocate(hio_labels(nvar,nx))
  hio_labels = 0
  !=====================================================
  ! Compute Gauss-Legendre quadrature points and weights
  !=====================================================
  call gl_quadrature(chsi_quad,w_quad,nquad)

  ! Mesh size
  dx=boxlen/dble(nx)

  !===========================
  ! Compute initial conditions
  !===========================
  do icell=1,nx
     xcell=(dble(icell)-0.5)*dx
     ! Loop over modes coefficients
     do i=1,n
        u(1:nvar,i,icell)=0.0
        ! Loop over quadrature points
        do j=1,nquad
           ! Quadrature point in physical space
           x_quad=xcell+dx/2.0*chsi_quad(j)
           call condinit(x_quad,uu)
           ! Perform integration using GL quadrature
           u(1:nvar,i,icell)=u(1:nvar,i,icell)+0.5* &
                & uu(1:nvar)* &
                & legendre(chsi_quad(j),i-1)* &
                & w_quad(j)
        end do
     end do
  end do

  !===============
  ! Main time loop
  !===============
  t=0
  iter=0
  counting = 0
  accum_time = 0
  !do while(iter<2)

  write(*,*) 'snap at:', t
       ureal = 0.0
       ! take snapshot
       do icell=1,nx
         do i = 1,n
           do j = 1,nquad
             ureal(1:nvar,i,icell) = ureal(1:nvar,i,icell) + u(1:nvar,j,icell)*legendre(chsi_quad(i),j-1)
           end do
         end do
       end do

       write(filename,"(A,A,A,I5.5)")base_folder,TRIM(title),"_nodes_",counting
       open(10,file=TRIM(filename)//".dat",form='formatted')
       do icell=1,nx
         do i = 1,n

          xcell = (dble(icell)-0.5)*dx + (dx/2.)*chsi_quad(i)
          write(10,'(7(1PE12.5,1X))')xcell,(ureal(ivar,i,icell),ivar=1,nvar)
        end do
       end do
       close(10)

       write(filename,"(A,A,A,I5.5)")base_folder,TRIM(title),"_modes_",counting
       open(11,file=TRIM(filename)//".dat",form='formatted')
       do icell=1,nx
         !do i = 1,n
          xcell = (dble(icell)-0.5)*dx
          write(11,'(15(1PE12.5,1X))')xcell,((u(ivar,i,icell),ivar=1,nvar),i=1,n)!,(u(ivar,2,icell),ivar=1,nvar)
        !end do
       end do
       close(11)

  call cpu_time(start)
  do while(t < tend)

     ! Compute time step
     call compute_max_speed(u,cmax)
     dt=0.2*dx/cmax/(2.0*dble(n)+1.0)
     dt = min(dt,tend-t)
     if(integrator=='RK1')then
        call compute_update(u,dudt)
        u=u+dt*dudt
     endif

     if(integrator=='RK2')then
        call compute_update(u,dudt)
        w1=u+dt*dudt
        call limiter(w1)
        call compute_update(w1,dudt)
        u=0.5*u+0.5*w1+0.5*dt*dudt
        call limiter(u)
     endif

     if(integrator=='RK3')then
        call compute_update(u,dudt)
        w1=u+dt*dudt
        call limiter(w1)
        call compute_update(w1,dudt)
        w2=0.75*u+0.25*w1+0.25*dt*dudt
        call limiter(w2)
        call compute_update(w2,dudt)
        u=1.0/3.0*u+2.0/3.0*w2+2.0/3.0*dt*dudt
        call limiter(u)
     endif


    if(integrator=='RA3')then
             call compute_update(u,dudt)
             f1 = dudt
             f2 = u + dt*f1
             !w1=u+dt*dudt
             call limiter(f2)
             call compute_update(f2,dudt)
             f2 = dudt
             f3 = u + dt*f1*1./4. + 1./4.*f2*dt
             call limiter(f3)
             call compute_update(f3,dudt)
             f3 = dudt
             u = u + 1.0/6.0*f1*dt + 1.0/6.0*f2*dt + 2./3.*f3*dt
             call limiter(u)
    endif

     if(integrator=='RK4')then
        call compute_update(u,dudt)
        w1=u+0.391752226571890*dt*dudt
        call limiter(w1)
        call compute_update(w1,dudt)
        w2=0.444370493651235*u+0.555629506348765*w1+0.368410593050371*dt*dudt
        call limiter(w2)
        call compute_update(w2,dudt)
        w3=0.620101851488403*u+0.379898148511597*w2+0.251891774271694*dt*dudt
        call limiter(w3)
        call compute_update(w3,dudt)
        w4=0.178079954393132*u+0.821920045606868*w3+0.544974750228521*dt*dudt
        !u=0.517231671970585*w2+0.096059710526147*w3+0.063692468666290*dt*dudt
        u= 0.00683325884039*u + 0.51723167208978*w2+0.12759831133288*w3+&
           &0.08460416338212*dt*dudt + 0.34833675773694*w4
        call limiter(w4)
        call compute_update(w4,dudt)
        u=u+0.22600748319395*dt*dudt
        call limiter(u)
     endif

     if(integrator=='RK5')then
       call compute_update(u,dudt)
       f1 = dudt
       f2 = u + f1*dt*0.392382208054010
       call limiter(f2)
       call compute_update(f2,dudt)
       f2 = dudt
       f3 = u + f1*dt*0.310348765296963 + f2*dt*0.523846724909595
       call limiter(f3)
       call compute_update(f3,dudt)
       f3 = dudt
       f4 = u + 0.114817342432177*f1*dt + 0.248293597111781*f2*dt
       call limiter(f4)
       call compute_update(f4,dudt)
       f4 = dudt
       f5 = u + 0.136041285050893*f1*dt + 0.163250087363657*f2*dt + 0.0*f3*dt + 0.557898557725281*f4*dt
       call limiter(f5)
       call compute_update(f5,dudt)
       f5 = dudt
       f6 = u + 0.135252145083336*f1*dt + 0.207274083097540*f2*dt - 0.180995372278096*f3*dt +&
       & 0.326486467604174*f4*dt + 0.348595427190109*f5*dt
       call limiter(f6)
       call compute_update(f6,dudt)
       f6 = dudt
       f7 = u + 0.082675687408986*f1*dt + 0.146472328858960*dt*f2 - 0.160507707995237*dt*f3 +&
       & 0.161924299217425*f4*dt + 0.028864227879979*dt*f5 + 0.070259587451358*f6*dt
       call limiter(f7)
       call compute_update(f7, dudt)
       f7 = dudt
       u = u + dt*f1*0.110184169931401 + dt*f2*0.122082833871843 - dt*0.117309105328437*f3 + &
       & 0.169714358772186*dt*f4 + 0.143346980044187*dt*f5 + 0.348926696469455*dt*f6 + &
       & 0.223054066239366*dt*f7
       call limiter(u)
     end if

    if(integrator=='RKF')then
         call compute_update(u,dudt)
         w1 = u + 16./135.*dt*dudt
         call limiter(w1)
         call compute_update(w1,dudt)
         w2 = 1./4.*u + 0.0*dudt*dt
         call limiter(w2)
         call compute_update(w2,dudt)
         w3 = 3./32.*u + 9./32.*w1 + 6656./12825.*dudt*dt
         call limiter(w3)
         call compute_update(w3,dudt)
         w4 = 1932./2197.*u - 7200./2197.*w1 + 7296./2197.*w2 + 28561./56430.*dt*dudt
         call limiter(w4)
         call compute_update(w4,dudt)
         w5 = 439./216.*u - 8.*w1 + 3680./513.*w2 - 845./4104.*w3 - 9./50.*dudt*dt
         call limiter(w5)
         call compute_update(w5,dudt)
         u = -8./27.*u + 2.*w1 - 3544./2565.*w2 + 1859./4104.*w3 &
         & - 11./40.*w4 + 2./55.*dudt*dt
         !w6 = -8./27.*u + 2.*w1 - 3544./2565.*w2 + 1859./4104.*w3 &
         !& - 11./40.*w4 + 2./55.*dudt*dt
         call limiter(u)
       endif

       if(integrator=='RKX')then
            call compute_update(u,dudt)
            w1 = u + 1./4.*dt*dudt
            call limiter(w1)
            call compute_update(w1,dudt)
            w2 = u + 3./32.*w1 + 9./32.*dudt*dt
            call limiter(w2)
            call compute_update(w2,dudt)
            w3 = u + 1932./2197*w1 - 7200./2197.*w2 + 7296./2197.*dudt*dt
            call limiter(w3)
            call compute_update(w3,dudt)
            w4 = 1932./2197.*u - 7200./2197.*w1 + 7296./2197.*w2 + 28561./56430.*dt*dudt
            call limiter(w4)
            call compute_update(w4,dudt)
            w5 = 439./216.*u - 8.*w1 + 3680./513.*w2 - 845./4104.*w3 - 9./50.*dudt*dt
            call limiter(w5)
            call compute_update(w5,dudt)
            u = -8./27.*u + 2.*w1 - 3544./2565.*w2 + 1859./4104.*w3 &
            & - 11./40.*w4 + 2./55.*dudt*dt
            call limiter(u)
          endif

     t=t+dt

     iter=iter+1
     write(*,*)'time=',iter,t,dt
     if (t > accum_time ) then
       counting = counting + 1
       write(*,*) 'snap at:', t
       !ureal = 0.0
       ! take snapshot
       !do icell=1,nx
       !  do i = 1,n
       !   do j = 1,nquad
       !      ureal(1:nvar,i,icell) = ureal(1:nvar,i,icell) + u(1:nvar,j,icell)*legendre(chsi_quad(i),j-1)
       !    end do
       !  end do
       !end do

       !write(filename,"(A,A,A,I5.5)")base_folder,TRIM(title),"_nodes_",counting
       !open(10,file=TRIM(filename)//".dat",form='formatted')
       !do icell=1,nx
       !  do i = 1,n

       !   xcell = (dble(icell)-0.5)*dx + (dx/2.)*chsi_quad(i)
       !   write(10,'(7(1PE12.5,1X))')xcell,(ureal(ivar,i,icell),ivar=1,nvar)
       ! end do
       !end do
       !close(10)

       !write(filename,"(A,A,A,I5.5)")base_folder,TRIM(title),"_modes_",counting
       !open(11,file=TRIM(filename)//".dat",form='formatted')
       !do icell=1,nx
         !do i = 1,n
        !  xcell = (dble(icell)-0.5)*dx
        !  write(11,'(15(1PE12.5,1X))')xcell,((u(ivar,i,icell),ivar=1,nvar),i=1,n)!,(u(ivar,2,icell),ivar=1,nvar)
        !end do
        !end do
        !close(11)

        call make_outputs(u, counting)

        accum_time = accum_time + freq_dt
     end if

  enddo
  call cpu_time(end)
  counting = counting + 1
  ! Reconstruct real solution
  !do icell=1,nx
  !  do i = 1,n
  !    do j = 1,nquad
  !      ureal(1:nvar,icell,i) = ureal(1:nvar,icell,i) + u(1:nvar,icell,j)*legendre(chsi_quad(i),j-1)
  !    end do
  !  end do
  !end do

  write(*,*) 'snap at:', t
       ureal = 0.0
       ! take snapshot
       do icell=1,nx
         do i = 1,n
           do j = 1,nquad
             ureal(1:nvar,i,icell) = ureal(1:nvar,i,icell) + u(1:nvar,j,icell)*legendre(chsi_quad(i),j-1)
           end do
         end do
       end do

       !write(filename,"(A,A,A,I5.5)")base_folder,TRIM(title),"_nodes_",counting
       !open(10,file=TRIM(filename)//".dat",form='formatted')
       !do icell=1,nx
       !  do i = 1,n
       !   xcell = (dble(icell)-0.5)*dx + (dx/2.)*chsi_quad(i)
       !   write(10,'(7(1PE12.5,1X))')xcell,(ureal(ivar,i,icell),ivar=1,nvar)
       ! end do
       !end do
       !close(10)

       !write(filename,"(A,A,A,I5.5)")base_folder,TRIM(title),"_modes_",counting
       !open(11,file=TRIM(filename)//".dat",form='formatted')
       !do icell=1,nx
         !do i = 1,n
        !  xcell = (dble(icell)-0.5)*dx
        !  write(11,'(15(1PE12.5,1X))')xcell,((u(ivar,i,icell),ivar=1,nvar),i=1,n)!,(u(ivar,2,icell),ivar=1,nvar)
        !end do
       !end do
       !close(11)

       call make_outputs(u, counting)

  write(*,*)'========================================'
  write(*,*)'time=',t,dt
  do icell=1,nx
     xcell=(dble(icell)-0.5)*dx
     call compute_primitive(u(1:nvar,1,icell),ww,gamma,nvar)
     write(*,'(7(1PE12.5,1X))')xcell,(ww(ivar),ivar=1,nvar)
  end do
  write(*,*) "Time = ",end-start," seconds."
  write(*,*) "Average iteration = " ,(end-start)/dble(iter), " seconds."

end program dg

subroutine make_outputs(fields,count)
  use dg_commons
  implicit none
  real(kind=8),dimension(1:nvar,1:n,1:nx)::fields
  integer::count, i, ivar, icell
  real(kind=8)::xcell, dx
  character(len=100)::filename

  dx = boxlen/dble(nx)

  write(filename,"(A,A,A,I5.5)")base_folder,TRIM(title),"_modes_",count
  open(11,file=TRIM(filename)//".dat",form='formatted')
  open(12,file=TRIM(filename)//"test.dat",form='formatted')
  do icell=1,nx
    xcell = (dble(icell)-0.5)*dx
    write(11,'(15(1PE12.5,1X))')xcell,((fields(ivar,i,icell),ivar=1,nvar),i=1,n)!,(hio_labels(ivar,icell),ivar=1,nvar)
    write(12,'(15(1PE12.5,1X))')xcell,((fields(ivar,i,icell),ivar=1,nvar),i=1,n),(hio_labels(ivar,icell),ivar=1,nvar)

  end do
  close(11)
  close(12)

end subroutine

subroutine read_params
  use dg_commons
  integer::i!,narg,iargc
  !character(LEN=80)::infile, info_file
  character(LEN=32)::arg
  !namelist/run_params/nx,ny,m,rk,boxlen_x,boxlen_y,gamma,cfl,tend
  !namelist/setup_params/bc,ninit
  !namelist/output_params/dtop,folder,dt_mon
  !namelist/source_params/xs,ys,rp,r_in,r_out,smp,Ms,Mp,grav
  ! Read namelist filename from command line argument
  !narg = iargc()
  !if(narg .lt. 1)then
  !   write(*,*)'File input.nml should contain a parameter namelist'
  !   stop
  !else
  !DO i = 1, iargc()
  !  CALL getarg(i, arg)
  !  WRITE (*,*) arg
  !END DO
  !n,nx,nquad,ninit,bc
  call getarg(1,arg)
  read(arg,*) nx

  call getarg(2,arg)
  read(arg,*) n

  call getarg(3,arg)
  read(arg,*) nquad

  call getarg(4,arg)
  read(arg,*) ninit

  call getarg(5,arg)
  read(arg,*) bc

  call getarg(6,arg)
  read(arg,*) limiter_type

  call getarg(7,arg)
  read(arg,*) title

  write(*,*) n,nx,nquad,ninit,bc
  !end if

  !namelist_file=TRIM(infile)

  !INQUIRE(file=infile,exist=nml_ok)
  !if(.not. nml_ok)then
  !   if(id==0)then
  !      write(*,*)'File '//TRIM(infile)//' does not exist'
  !   endif
  !   stop
  !end if

end subroutine read_params

subroutine get_modes(u,um)
  use dg_commons
  integer:: icell, i, j
  real(kind=8)::xcell, x_quad, legendre
  real(kind=8),dimension(1:nvar)::uu
  real(kind=8),dimension(1:nvar,n,nx)::u, um

  do icell=1,nx
     !xcell=(dble(icell)-0.5)*dx
     ! Loop over modes coefficients
     do i=1,n
        u(1:nvar,i,icell)=0.0
        ! Loop over quadrature points
        do j=1,nquad
           ! Quadrature point in physical space
           !x_quad=xcell+dx/2.0*chsi_quad(j)
           !call condinit(x_quad,uu)
           ! Perform integration using GL quadrature
           u(1:nvar,i,icell)=u(1:nvar,i,icell)+0.5* &
                & uu(1:nvar)* &
                & legendre(chsi_quad(j),i-1)* &
                & w_quad(j)
        end do
     end do
  end do

end subroutine get_modes

subroutine get_nodes(um,u)
  use dg_commons
  integer:: icell, i, j
  real(kind=8)::xcell, x_quad, legendre
  real(kind=8),dimension(1:nvar,n,nx)::u, um
  do icell=1,nx
    do i = 1,n
      do j = 1,nquad
        u(1:nvar,i,icell) = u(1:nvar,i,icell) + um(1:nvar,j,icell)*legendre(chsi_quad(i),j-1)
      end do
    end do
  end do

end subroutine get_nodes

!==============================================
subroutine limiter(u)
  use dg_commons
  implicit none
  real(kind=8),dimension(1:nvar,1:n,1:nx)::u
  !================================================================
  ! This routine applies the moment limiter to the current solution
  ! as in Krivodonova, 2007, JCP, 226, 879
  ! using characteristic variables.
  !================================================================
  real(kind=8),dimension(1:nvar,1:n,1:nx)::u_lim, u_lim_nodes
  real(kind=8),dimension(1:nvar,1:n)::w_lim
  real(kind=8),dimension(1:nvar,1:n)::wL,wM,wR
  real(kind=8),dimension(1:nvar)::w
  real(kind=8),dimension(1:nvar)::uL,uM,uR
  real(kind=8),dimension(1:nvar)::u_left,u_right,w_left,w_right
  real(kind=8)::minmod,maxmod
  real(kind=8)::switch_left,switch_right
  real(kind=8)::u_min,u_max,w_min
  real(kind=8)::coeff_i,coeff_ip1,coeff,D2u
  integer::icell,i,j,iface,ileft,iright,ivar
  if(n==1)return
  ! Compute classical minmod limiter
  u_lim = u
  if(use_limiter)then
    !call limiter_in_c(u_lim)
    !call limiter_tvd(u_lim,0.0)
    if (limiter_type == 'MIN') then
      M = 0.0
      call limiter_tvd(u_lim,M)
    else if (limiter_type == 'TVD') then
      call limiter_tvd(u_lim,M)
    else if (limiter_type =='HIO') then
      call limiter_hio(u_lim)
    else if (limiter_type == 'NNM') then 
      call limiter_ml1(u_lim)
    !else if (limiter_type == 'NNH') then
    !  call limiter_ml2(u_lim)
    end if
  end if

  !write(*,*) u
  !write(*,*) u_lim
  !write(*,*) u_lim - 2*u

  u = u_lim
end subroutine limiter


subroutine limiter_ml1(u)
  use dg_commons
  implicit none
  real(kind=8),dimension(1:nvar,1:n,1:nx)::u
  !================================================================
  ! This routine applies the moment limiter to the current solution
  ! as in Krivodonova, 2007, JCP, 226, 879
  ! using characteristic variables.
  !================================================================
  real(kind=8),dimension(1:nvar,1:n,1:nx)::u_lim, w, un, wn
  real(kind=8),dimension(1:nvar,1:n)::w_lim
  real(kind=8),dimension(1:nvar,1:n)::wL,wM,wR
  real(kind=8),dimension(1:nvar)::uL,uM,uR
  real(kind=8),dimension(1:nvar)::u_left,u_right,w_left,w_right
  real(kind=8)::switch_left,switch_right, minmod, dx, w_min
  integer::icell,i,j,iface,ileft,iright,ivar,jvar
  integer,dimension(1:nvar)::label
  integer::limiting_ml, lab

  if(n==1)return

  ! Compute classical minmod limiter
  u_lim=u

  do icell=1,nx
     ileft=icell-1
     iright=icell+1
     switch_left=1.0
     switch_right=1.0
     if(bc==1)then
        if(icell==1)ileft=nx
        if(icell==nx)iright=1
     endif
     if(bc==2)then
        if(icell==1)ileft=1
        if(icell==nx)iright=nx
     endif
     if(bc==3)then
        if(icell==1)then
           ileft=1
           switch_left=-1.0
        endif
        if(icell==nx)then
           iright=nx
           switch_right=-1.0
        endif
     endif

     ! shock detection
     call get_nodes(u,un)
     call compute_primitive(un,wn,gamma,nvar)
     call get_modes(wn,w)
     w=u
     !call compute_primitive(u(1:nvar,1,icell),w,gamma,nvar)
     !do jvar = 1,nvar
       call detect_shock(w(jvar,:,icell), w(jvar,:,iright), w(jvar,:,ileft),label) ! gets rid of having to use parameters from ml
       !write(*,*) label
       limiting_ml = 0
       !do lab = 1,nvar
         if (label(1) == 1) then
            limiting_ml = 1
            write(*,*) 'limiting'
         end if
       !end do

       if (limiting_ml == 1) then
         ! apply TVD limiter
         !write(*,*) 'gonna limit'
         dx = boxlen/dble(nx)
         i = 1
         w=0
         call compute_primitive(u(1:nvar,1,icell),w,gamma,nvar)
         ! Loop only over high-order modes
         uL(1:nvar)=(u(1:nvar,i,icell)-u(1:nvar,i,ileft))
         uR(1:nvar)=(u(1:nvar,i,iright)-u(1:nvar,i,icell))
         uM(1:nvar)=u(1:nvar,i+1,icell)
         uL(2)=switch_left*uL(2)
         uR(2)=switch_right*uR(2)
         call cons_to_char(uL,wL(1:nvar,i+1),w,nvar,gamma)
         call cons_to_char(uR,wR(1:nvar,i+1),w,nvar,gamma)
         call cons_to_char(uM,wM(1:nvar,i+1),w,nvar,gamma)
         w_lim=wM
         ! Loop over variables
         !do ivar=1,nvar
         ivar = 1
            !if (abs(wM(ivar,2)).LE.(M*dx**2)) then
            !  wM(ivar,2) = wM(ivar,2)
            !else
              w_min=minmod(wL(ivar,2),wM(ivar,2),wR(ivar,2))
              w_lim(ivar,2)=w_min
              if(ABS(w_min-wM(ivar,2)).GT.0.01*ABS(wM(ivar,2))) then
                wM(ivar,2) = 0.0
              end if
            !end if
         !end do
         ! End loop over variables
         ! Compute conservative variables
         ! Loop only over high-order modes
         !i = 1
         call char_to_cons(w_lim(1:nvar,i+1),u_lim(1:nvar,i+1,icell),w,nvar,gamma)

       else
         ! pass
       end if
     !end do

   end do

  ! Check for unphysical values in the limited states
  !do icell=1,nx
     ! Compute primitive variable
     !call compute_primitive(u_lim(1:nvar,1,icell),w,gamma,nvar)
     !u_left(1:nvar)=0.0; u_right(1:nvar)=0.0
     ! Loop over modes
     !do i=1,n
     !   u_left(1:nvar)=u_left(1:nvar)+u_lim(1:nvar,i,icell)*(-1.0)**(i-1)*sqrt(2.0*dble(i)-1.0)
     !   u_right(1:nvar)=u_right(1:nvar)+u_lim(1:nvar,i,icell)*sqrt(2.0*dble(i)-1.0)
     !nd do
     !call cons_to_prim(u_left,w_left,w,nvar,gamma)
     !call cons_to_prim(u_right,w_right,w,nvar,gamma)
     !if(w_left(1)<1d-10.OR.w_right(1)<1d-10.OR.w_left(3)<1d-10.OR.w_left(3)<1d-10)then
     !   u_lim(1:nvar,2:n,icell)=0.0
     !endif
  !end do

   u = u_lim
end subroutine limiter_ml1


subroutine limiter_ml2(u)
  use dg_commons
  implicit none
  real(kind=8),dimension(1:nvar,1:n,1:nx)::u
  !================================================================
  ! This routine applies the moment limiter to the current solution
  ! as in Krivodonova, 2007, JCP, 226, 879
  ! using characteristic variables.
  !================================================================
  real(kind=8),dimension(1:nvar,1:n,1:nx)::u_lim, w, un, wn
  real(kind=8),dimension(1:nvar,1:n)::w_lim
  real(kind=8),dimension(1:nvar,1:n)::wL,wM,wR
  real(kind=8),dimension(1:nvar)::uL,uM,uR
  real(kind=8),dimension(1:nvar)::u_left,u_right,w_left,w_right
  real(kind=8)::switch_left,switch_right, minmod, dx, w_min
  integer::icell,i,j,iface,ileft,iright,ivar,jvar
  integer::label
  if(n==1)return

  ! Compute classical minmod limiter
  u_lim=u

  do icell=1,nx
     ileft=icell-1
     iright=icell+1
     switch_left=1.0
     switch_right=1.0
     if(bc==1)then
        if(icell==1)ileft=nx
        if(icell==nx)iright=1
     endif
     if(bc==2)then
        if(icell==1)ileft=1
        if(icell==nx)iright=nx
     endif
     if(bc==3)then
        if(icell==1)then
           ileft=1
           switch_left=-1.0
        endif
        if(icell==nx)then
           iright=nx
           switch_right=-1.0
        endif
     endif

     ! shock detection
     call get_nodes(u,un)
     call compute_primitive(un,wn,gamma,nvar)
     call get_modes(wn,w)
     !call compute_primitive(u(1:nvar,1,icell),w,gamma,nvar)
     do jvar = 1,nvar
       call detect_shock(w(jvar,:,icell), w(jvar,:,iright), w(jvar,:,ileft),label) ! gets rid of having to use parameters from ml
       !write(*,*) label
       if (label == 1) then
         ! apply TVD limiter
         !write(*,*) 'gonna limit'
         dx = boxlen/dble(nx)
         i = 1
         call compute_primitive(u(1:nvar,1,icell),w,gamma,nvar)
         ! Loop only over high-order modes
         uL(1:nvar)=(u(1:nvar,i,icell)-u(1:nvar,i,ileft))
         uR(1:nvar)=(u(1:nvar,i,iright)-u(1:nvar,i,icell))
         uM(1:nvar)=u(1:nvar,i+1,icell)
         uL(2)=switch_left*uL(2)
         uR(2)=switch_right*uR(2)
         call cons_to_char(uL,wL(1:nvar,i+1),w,nvar,gamma)
         call cons_to_char(uR,wR(1:nvar,i+1),w,nvar,gamma)
         call cons_to_char(uM,wM(1:nvar,i+1),w,nvar,gamma)
         w_lim=wM
         ! Loop over variables
         do ivar=1,nvar
            !if (abs(wM(ivar,2)).LE.(M*dx**2)) then
            !  wM(ivar,2) = wM(ivar,2)
            !else
              w_min=minmod(wL(ivar,2),wM(ivar,2),wR(ivar,2))
              w_lim(ivar,2)=w_min
              if(ABS(w_min-wM(ivar,2)).GT.0.01*ABS(wM(ivar,2))) then
                wM(ivar,2) = 0.0
              end if
            !end if
         end do
         ! End loop over variables
         ! Compute conservative variables
         ! Loop only over high-order modes
         i = 1
         call char_to_cons(w_lim(1:nvar,i+1),u_lim(1:nvar,i+1,icell),w,nvar,gamma)

       else
         ! pass
       end if
     end do

   end do
   u = u_lim
end subroutine limiter_ml2

subroutine limiter_hio(u)
  use dg_commons
  implicit none
  real(kind=8),dimension(1:nvar,1:n,1:nx)::u
  !================================================================
  ! This routine applies the moment limiter to the current solution
  ! as in Krivodonova, 2007, JCP, 226, 879
  ! using characteristic variables.
  !================================================================
    real(kind=8),dimension(1:nvar,1:n,1:nx)::u_lim, u_lim_nodes
  real(kind=8),dimension(1:nvar,1:n)::w_lim
  real(kind=8),dimension(1:nvar,1:n)::wL,wM,wR
  real(kind=8),dimension(1:nvar)::w
  real(kind=8),dimension(1:nvar)::uL,uM,uR
  real(kind=8),dimension(1:nvar)::u_left,u_right,w_left,w_right, u_left_original, u_right_original
  real(kind=8)::minmod,maxmod
  real(kind=8)::switch_left,switch_right
  real(kind=8)::u_min,u_max,w_min
  real(kind=8)::coeff_i,coeff_ip1,coeff,D2u
  integer::icell,i,j,iface,ileft,iright,ivar, limited
  if(n==1)return
  ! Compute classical minmod limiter
  u_lim=u
  if(use_limiter)then
  hio_labels = 0
  do icell=1,nx
     ileft=icell-1
     iright=icell+1
     switch_left=1.0
     switch_right=1.0
     if(bc==1)then
        if(icell==1)ileft=nx
        if(icell==nx)iright=1
     endif
     if(bc==2)then
        if(icell==1)ileft=1
        if(icell==nx)iright=nx
     endif
     if(bc==3)then
        if(icell==1)then
           ileft=1
           switch_left=-1.0
        endif
        if(icell==nx)then
           iright=nx
           switch_right=-1.0
        endif
     endif
     ! Compute primitive variable for all modes
     call compute_primitive(u(1:nvar,1,icell),w,gamma,nvar)
     ! Loop only over high-order modes
     do i=n-1,1,-1
        ! Renormalise to get proper Legendre polynomials
        ! and corresponding derivatives
        coeff_i=sqrt(2.0*dble(i-1)+1.0)*(2.0*dble(i)-1)
        coeff_ip1=sqrt(2.0*dble(i)+1.0)*(2.0*dble(i)-1)
        uL(1:nvar)=(u(1:nvar,i,icell)-u(1:nvar,i,ileft))*coeff_i/coeff_ip1
        uR(1:nvar)=(u(1:nvar,i,iright)-u(1:nvar,i,icell))*coeff_i/coeff_ip1
        uM(1:nvar)=u(1:nvar,i+1,icell)
        uL(2)=switch_left*uL(2)
        uR(2)=switch_right*uR(2)
        call cons_to_char(uL,wL(1:nvar,i+1),w,nvar,gamma)
        call cons_to_char(uR,wR(1:nvar,i+1),w,nvar,gamma)
        call cons_to_char(uM,wM(1:nvar,i+1),w,nvar,gamma)
     end do
     w_lim=wM
     ! Loop over variables
     do ivar=1,nvar
        ! Loop only over high-order modes
        limited = 0
        do i=n-1,1,-1
           w_min=minmod(wL(ivar,i+1),wM(ivar,i+1),wR(ivar,i+1))
           w_lim(ivar,i+1)=w_min
           if(ABS(w_min-wM(ivar,i+1)).LT.0.05*ABS(wM(ivar,i+1))) then
           !if(ABS(w_min-wM(ivar,i+1)).LT.0.0001) then
            exit
           else 
            !write(*,*) 'limited'
            !hio_labels(ivar,icell) = 1
           end if
        end do
        ! End loop over modes
     end do
     ! End loop over variables
     ! Compute conservative variables
     ! Loop only over high-order modes
     do i=n-1,1,-1
        call char_to_cons(w_lim(1:nvar,i+1),u_lim(1:nvar,i+1,icell),w,nvar,gamma)
     end do
  end do
  ! End loop over cells
  endif

  ! Check for unphysical values in the limited states
  do icell=1,nx
     ! Compute primitive variable
     call compute_primitive(u_lim(1:nvar,1,icell),w,gamma,nvar)
     u_left(1:nvar)=0.0; u_right(1:nvar)=0.0
     u_left_original = 0.0; u_right_original = 0.0
     ! Loop over modes
     do i=1,n
        u_left(1:nvar)=u_left(1:nvar)+u_lim(1:nvar,i,icell)*(-1.0)**(i-1)*sqrt(2.0*dble(i)-1.0)
        u_right(1:nvar)=u_right(1:nvar)+u_lim(1:nvar,i,icell)*sqrt(2.0*dble(i)-1.0)
        u_left_original(1:nvar) = u_left_original(1:nvar) + u(1:nvar,i,icell)*(-1.0)**(i-1)*sqrt(2.0*dble(i)-1.0)
        u_right_original(1:nvar) = u_right_original(1:nvar) + u(1:nvar,i,icell)*sqrt(2.0*dble(i)-1.0)
     end do
     call cons_to_prim(u_left,w_left,w,nvar,gamma)
     call cons_to_prim(u_right,w_right,w,nvar,gamma)
     if(w_left(1)<1d-10.OR.w_right(1)<1d-10.OR.w_left(3)<1d-10.OR.w_left(3)<1d-10)then
        u_lim(1:nvar,2:n,icell)=0.0
     endif
     
     do ivar= 1,nvar
      if ((abs(u_left(ivar)-u_left_original(ivar)).gt.(0.0005*abs(u_left_original(ivar))))&
      &.or.(abs(u_right(ivar)-u_right_original(ivar)).gt.(0.0005*abs(u_right_original(ivar))))) then
        write(*,*) u_left(ivar),u_left_original(ivar)
        hio_labels(ivar,icell) = 1
      end if
     end do
     
  end do

  ! Update variables with limited states
  u=u_lim
end subroutine limiter_hio

subroutine limiter_tvd(u,Mparam)
  use dg_commons
  implicit none
  real(kind=8),dimension(1:nvar,1:n,1:nx)::u
  real(kind=8):: dx, Mparam
  !================================================================
  ! This routine applies the moment limiter to the current solution
  ! as in Krivodonova, 2007, JCP, 226, 879
  ! using characteristic variables.
  !================================================================
  real(kind=8),dimension(1:nvar,1:n,1:nx)::u_lim, u_lim_nodes
  real(kind=8),dimension(1:nvar,1:n)::w_lim
  real(kind=8),dimension(1:nvar,1:n)::wL,wM,wR
  real(kind=8),dimension(1:nvar)::w
  real(kind=8),dimension(1:nvar)::uL,uM,uR
  real(kind=8),dimension(1:nvar)::u_left,u_right,w_left,w_right
  real(kind=8)::minmod,maxmod
  real(kind=8)::switch_left,switch_right
  real(kind=8)::u_min,u_max,w_min
  real(kind=8)::coeff_i,coeff_ip1,coeff,D2u
  integer::icell,i,j,iface,ileft,iright,ivar
  dx  = boxlen/dble(nx)
  if(n==1)return
  ! Compute classical minmod limiter
  u_lim=u
  do icell=1,nx
     ileft=icell-1
     iright=icell+1
     switch_left=1.0
     switch_right=1.0
     if(bc==1)then
        if(icell==1)ileft=nx
        if(icell==nx)iright=1
     endif
     if(bc==2)then
        if(icell==1)ileft=1
        if(icell==nx)iright=nx
     endif
     if(bc==3)then
        if(icell==1)then
           ileft=1
           switch_left=-1.0
        endif
        if(icell==nx)then
           iright=nx
           switch_right=-1.0
        endif
     endif
     ! Compute primitive variable for all modes
     call compute_primitive(u(1:nvar,1,icell),w,gamma,nvar)
     ! Loop only over high-order modes
     do i=1,1!n-1,1,-1
        ! Renormalise to get proper Legendre polynomials
        ! and corresponding derivatives
        coeff_i=sqrt(2.0*dble(i-1)+1.0)*(2.0*dble(i)-1)
        coeff_ip1=sqrt(2.0*dble(i)+1.0)*(2.0*dble(i)-1)
        uL(1:nvar)=(u(1:nvar,i,icell)-u(1:nvar,i,ileft))*coeff_i/coeff_ip1
        uR(1:nvar)=(u(1:nvar,i,iright)-u(1:nvar,i,icell))*coeff_i/coeff_ip1
        uM(1:nvar)=u(1:nvar,i+1,icell)
        uL(2)=switch_left*uL(2)
        uR(2)=switch_right*uR(2)
        call cons_to_char(uL,wL(1:nvar,i+1),w,nvar,gamma)
        call cons_to_char(uR,wR(1:nvar,i+1),w,nvar,gamma)
        call cons_to_char(uM,wM(1:nvar,i+1),w,nvar,gamma)
     end do

     w_lim=wM
     ! Loop over variables
     do ivar=1,nvar
        ! Loop only over high-order modes
        !do i=n-1,1,-1
        if (abs(wM(ivar,2)).LE.(Mparam*dx**2)) then
          wM(ivar,2) = wM(ivar,2)
        else
          w_min=minmod(wL(ivar,2),wM(ivar,2),wR(ivar,2))
          w_lim(ivar,2)=w_min
          if(ABS(w_min-wM(ivar,2)).GT.0.01*ABS(wM(ivar,2))) then
            wM(ivar,2) = 0.0
          end if
        end if
        ! End loop over modes
     end do
     ! End loop over variables
     ! Compute conservative variables
     ! Loop only over high-order modes
     do i=1,1
        call char_to_cons(w_lim(1:nvar,i+1),u_lim(1:nvar,i+1,icell),w,nvar,gamma)
     end do
  end do
  ! End loop over cells

  ! Check for unphysical values in the limited states
  do icell=1,nx
     ! Compute primitive variable
     call compute_primitive(u_lim(1:nvar,1,icell),w,gamma,nvar)
     u_left(1:nvar)=0.0; u_right(1:nvar)=0.0
     ! Loop over modes
     do i=1,n
        u_left(1:nvar)=u_left(1:nvar)+u_lim(1:nvar,i,icell)*(-1.0)**(i-1)*sqrt(2.0*dble(i)-1.0)
        u_right(1:nvar)=u_right(1:nvar)+u_lim(1:nvar,i,icell)*sqrt(2.0*dble(i)-1.0)
     end do
     call cons_to_prim(u_left,w_left,w,nvar,gamma)
     call cons_to_prim(u_right,w_right,w,nvar,gamma)
     if(w_left(1)<1d-10.OR.w_right(1)<1d-10.OR.w_left(3)<1d-10.OR.w_left(3)<1d-10)then
        u_lim(1:nvar,2:n,icell)=0.0
     endif
  end do
  ! Update variables with limited states
  u=u_lim

end subroutine limiter_tvd

!==============================================
subroutine compute_update(u,dudt)
  use dg_commons
  implicit none
  real(kind=8),dimension(1:nvar,1:n,1:nx)::u,dudt
  !===========================================================
  ! This routine computes the DG update for the input state u.
  !===========================================================
  real(kind=8),dimension(1:nvar,1:nquad)::u_quad,flux_quad
  real(kind=8),dimension(1:nvar,1:n,1:nx)::flux_vol
  real(kind=8),dimension(1:nvar,1:nx)::u_left,u_right
  real(kind=8),dimension(1:nvar)::flux_riemann,u_tmp
  real(kind=8),dimension(1:nvar,1:nx+1)::flux_face

  integer::icell,i,j,iface,ileft,iright,ivar
  real(kind=8)::legendre,legendre_prime
  real(kind=8)::chsi_left=-1,chsi_right=+1
  real(kind=8)::dx,x_quad
  real(kind=8)::cmax,oneoverdx,c_left,c_right

  dx=boxlen/dble(nx)
  oneoverdx=1.0/dx

  ! Loop over cells
  do icell=1,nx

     !==================================
     ! Compute flux at quadrature points
     !==================================
     ! Loop over quadrature points
     do j=1,nquad
        u_quad(1:nvar,j)=0.0
        ! Loop over modes
        do i=1,n
           u_quad(1:nvar,j)=u_quad(1:nvar,j)+u(1:nvar,i,icell)*legendre(chsi_quad(j),i-1)
        end do
        ! Compute flux at quadrature points
        call compute_flux(u_quad(1:nvar,j),flux_quad(1:nvar,j),gamma,nvar)
     end do

     !================================
     ! Compute volume integral DG term
     !================================
     ! Loop over modes
     do i=1,n
        flux_vol(1:nvar,i,icell)=0.0
        ! Loop over quadrature points
        do j=1,nquad
           flux_vol(1:nvar,i,icell)=flux_vol(1:nvar,i,icell)+ &
                & flux_quad(1:nvar,j)* &
                & legendre_prime(chsi_quad(j),i-1)* &
                & w_quad(j)
        end do
     end do

     !==============================
     ! Compute left and right states
     !==============================
     u_left(1:nvar,icell)=0.0
     u_right(1:nvar,icell)=0.0
     ! Loop over modes
     do i=1,n
        u_left(1:nvar,icell)=u_left(1:nvar,icell)+u(1:nvar,i,icell)*legendre(chsi_left,i-1)
        u_right(1:nvar,icell)=u_right(1:nvar,icell)+u(1:nvar,i,icell)*legendre(chsi_right,i-1)
     end do
  end do
  ! End loop over cells

  !==========================================
  ! Compute physical flux from Riemann solver
  !==========================================
  ! Loop over faces
  do iface=1,nx+1
     ileft=iface-1
     iright=iface
     if(bc==1)then
        ! Periodic boundary conditions
        if(iface==1)ileft=nx
        if(iface==nx+1)iright=1
     endif
     if(bc==2.or.bc==3)then
        ! Zero gradient boundary conditions
        if(iface==1)ileft=1
        if(iface==nx+1)iright=nx
     endif
     ! Compute physical flux using Riemann solver
     select case(riemann)
     case(1)
        call riemann_llf(u_right(1:nvar,ileft),u_left(1:nvar,iright)&
             & ,flux_riemann,gamma,nvar)
     case(2)
        call riemann_hllc(u_right(1:nvar,ileft),u_left(1:nvar,iright)&
             & ,flux_riemann,gamma,nvar)
     end select
     flux_face(1:nvar,iface)=flux_riemann
     ! Compute boundary flux for reflexive BC
     if(bc==3.AND.iface==1)then
        u_tmp(1:nvar)=u_left(1:nvar,iright)
        u_tmp(2)=-u_tmp(2)
        select case(riemann)
        case(1)
           call riemann_llf(u_tmp,u_left(1:nvar,iright)&
                & ,flux_riemann,gamma,nvar)
        case(2)
           call riemann_hllc(u_tmp,u_left(1:nvar,iright)&
                & ,flux_riemann,gamma,nvar)
        end select
        flux_face(1:nvar,iface)=flux_riemann
     endif
     if(bc==3.AND.iface==nx+1)then
        u_tmp(1:nvar)=u_right(1:nvar,ileft)
        u_tmp(2)=-u_tmp(2)
        select case(riemann)
        case(1)
           call riemann_llf(u_right(1:nvar,ileft),u_tmp&
           &,flux_riemann,gamma,nvar)
        case(2)
           call riemann_hllc(u_right(1:nvar,ileft),u_tmp&
                &,flux_riemann,gamma,nvar)
        end select
        flux_face(1:nvar,iface)=flux_riemann
     endif
  end do

  !========================
  ! Compute final DG update
  !========================
  ! Loop over cells
  do icell=1,nx
     ! Loop over modes
     do i=1,n
        dudt(1:nvar,i,icell)=oneoverdx*(flux_vol(1:nvar,i,icell) &
             & -(flux_face(1:nvar,icell+1)*legendre(chsi_right,i-1) &
             & -flux_face(1:nvar,icell)*legendre(chsi_left,i-1)))
     end do
  end do
  !print*,dudt(1,:,:)
  !pause
  !call add_viscosity(dudt,u)

end subroutine compute_update
!==============================================
subroutine condinit(x,uu)
  use dg_commons
  real(kind=8)::x
  real(kind=8),dimension(1:nvar)::uu
  !==============================================
  ! This routine computes the initial conditions.
  !==============================================
  real(kind=8),dimension(1:nvar)::ww
  real(kind=8)::dpi=acos(-1d0)

  ! Compute primitive variables
  select case (ninit)
     case(1) ! sine wave (tend=1 or 10)
        ww(1)=1.0+0.5*sin(2.0*dpi*x)
        ww(2)=1.0
        ww(3)=1.0
     case(2) ! step function (tend=1 or 10)
        if(abs(x-0.5)<0.25)then
           ww(1)=4.
        else
           ww(1)=1.0
        endif
        ww(2)=1.0
        ww(3)=1.0
     case(3) ! gaussian + square pulses (tend=1 or 10)
        ww(1)=1.+exp(-(x-0.25)**2/2.0/0.05**2)
        if(abs(x-0.7)<0.1)then
           ww(1)=ww(1)+1.
        endif
        ww(2)=1.0
        ww(3)=1.0
     case(4) ! Sod test (tend=0.245)
        if(abs(x-0.25)<0.25)then
           ww(1)=1.0
           ww(2)=0.0
           ww(3)=1.0
        else
           ww(1)=0.125
           ww(2)=0.0
           ww(3)=0.1
        endif
     case(5) ! blast wave test  (tend=0.038)
        if(x<0.1)then
           ww(1)=1.0
           ww(2)=0.0
           ww(3)=1000.0
        else if(x<0.9)then
           ww(1)=1.0
           ww(2)=0.0
           ww(3)=0.01
        else
           ww(1)=1.0
           ww(2)=0.0
           ww(3)=100.
        endif
     case(6) ! shock entropy interaction (tend=2)
        if(x<10.0)then
           ww(1)=3.857143
           ww(2)=-0.920279
           ww(3)=10.333333
        else
           ww(1)=1.0+0.2*sin(5.0*(x-10.0))
           ww(2)=-3.549648
           ww(3)=1.0
        endif
    case(7)
        ! gaussian blob
        ww(1)=1.+3.*exp(-100.*(x-0.5)**2)
        ww(2)=1.0
        ww(3)=1.0

     end select

     ! Convert primitive to conservative
     uu(1)=ww(1)
     uu(2)=ww(1)*ww(2)
     uu(3)=ww(3)/(gamma-1.0)+0.5*ww(1)*ww(2)**2

  return
end subroutine condinit
!==============================================
subroutine compute_max_speed(u,cmax)
  use dg_commons
  implicit none
  real(kind=8),dimension(1:nvar,1:n,1:nx)::u
  real(kind=8)::cmax
  !==============================================
  ! This routine computes the maximum wave speed.
  !==============================================
  integer::icell
  real(kind=8)::speed
  ! Compute max sound speed
  cmax=0.0
  do icell=1,nx
     call compute_speed(u(1:nvar,1,icell),speed,gamma,nvar)
     cmax=MAX(cmax,speed)
  enddo
end subroutine compute_max_speed

!==============================================
function minmod(x,y,z)
  implicit none
  real(kind=8)::x,y,z,s,minmod
  s=sign(1d0,x)
  if(sign(1d0,y)==s.AND.sign(1d0,z)==s)then
     minmod=s*min(abs(x),abs(y),abs(z))
  else
     minmod=0.0
  endif
  return
end function minmod
!==============================================
subroutine compute_speed(u,speed,gamma,nvar)
  implicit none
  integer::nvar
  real(kind=8),dimension(1:nvar)::u
  real(kind=8)::speed,gamma
  real(kind=8),dimension(1:nvar)::w
  real(kind=8)::cs
  ! Compute primitive variables
  call compute_primitive(u,w,gamma,nvar)
  ! Compute sound speed
  cs=sqrt(gamma*max(w(3),1d-10)/max(w(1),1d-10))
  speed=abs(w(2))+cs
end subroutine compute_speed
!==============================================
subroutine compute_flux(u,flux,gamma,nvar)
  implicit none
  integer::nvar
  real(kind=8),dimension(1:nvar)::u
  real(kind=8),dimension(1:nvar)::flux
  real(kind=8)::gamma
  real(kind=8),dimension(1:nvar)::w
  ! Compute primitive variables
  call compute_primitive(u,w,gamma,nvar)
  ! Compute flux
  flux(1)=w(2)*u(1)
  flux(2)=w(2)*u(2)+w(3)
  flux(3)=w(2)*u(3)+w(3)*w(2)
end subroutine compute_flux
!==============================================
subroutine compute_primitive(u,w,gamma,nvar)
  implicit none
  integer::nvar
  real(kind=8)::gamma
  real(kind=8),dimension(1:nvar)::u
  real(kind=8),dimension(1:nvar)::w
  ! Compute primitive variables
  w(1)=u(1)
  w(2)=u(2)/w(1)
  w(3)=(gamma-1.0)*(u(3)-0.5*w(1)*w(2)**2)
end subroutine compute_primitive
!==============================================
subroutine cons_to_prim(du,dw,w,nvar,gamma)
  implicit none
  integer::nvar
  real(kind=8)::gamma
  real(kind=8),dimension(1:nvar)::du,dw,w
  dw(1)=du(1)
  dw(2)=(du(2)-w(2)*du(1))/w(1)
  dw(3)=(gamma-1.0)*(0.5*w(2)**2*du(1)-w(2)*du(2)+du(3))
end subroutine cons_to_prim
!==============================================
subroutine prim_to_cons(dw,du,w,nvar,gamma)
  implicit none
  integer::nvar
  real(kind=8)::gamma
  real(kind=8),dimension(1:nvar)::du,dw,w
  du(1)=dw(1)
  du(2)=w(2)*dw(1)+w(1)*dw(2)
  du(3)=0.5*w(2)**2*dw(1)+w(1)*w(2)*dw(2)+dw(3)/(gamma-1.0)
end subroutine prim_to_cons
!==============================================
subroutine cons_to_char(du,dw,w,nvar,gamma)
  implicit none
  integer::nvar
  real(kind=8)::gamma
  real(kind=8),dimension(1:nvar)::du,dw,w,dp
  real(kind=8)::csq,cs
  csq=gamma*max(w(3),1d-10)/max(w(1),1d-10)
  cs=sqrt(csq)
  call cons_to_prim(du,dp,w,nvar,gamma)
  dw(1)=dp(1)-dp(3)/csq
  dw(2)=0.5*(dp(3)/csq+dp(2)*w(1)/cs)
  dw(3)=0.5*(dp(3)/csq-dp(2)*w(1)/cs)
end subroutine cons_to_char
!==============================================
subroutine char_to_cons(dw,du,w,nvar,gamma)
  implicit none
  integer::nvar
  real(kind=8)::gamma
  real(kind=8),dimension(1:nvar)::du,dw,w,dp
  real(kind=8)::csq,cs
  csq=gamma*max(w(3),1d-10)/max(w(1),1d-10)
  cs=sqrt(csq)
  dp(1)=dw(1)+dw(2)+dw(3)
  dp(2)=(dw(2)-dw(3))*cs/w(1)
  dp(3)=(dw(2)+dw(3))*csq
  call prim_to_cons(dp,du,w,nvar,gamma)
end subroutine char_to_cons
!==============================================
subroutine cons_to_cons(du,dw,w,nvar,gamma)
  implicit none
  integer::nvar
  real(kind=8)::gamma
  real(kind=8),dimension(1:nvar)::du,dw,w,dp
  real(kind=8)::csq,cs
  dw(1)=du(1)
  dw(2)=du(2)
  dw(3)=du(3)
end subroutine cons_to_cons
!==============================================
subroutine riemann_llf(uleft,uright,fgdnv,gamma,nvar)
  implicit none
  integer::nvar
  real(kind=8)::gamma
  real(kind=8),dimension(1:nvar)::uleft,uright
  real(kind=8),dimension(1:nvar)::fgdnv
  real(kind=8)::cleft,cright,cmax
  real(kind=8),dimension(1:nvar)::fleft,fright
  ! Maximum wave speed
  call compute_speed(uleft,cleft,gamma,nvar)
  call compute_speed(uright,cright,gamma,nvar)
  cmax=max(cleft,cright)
  ! Compute flux at left and right points
  call compute_flux(uleft,fleft,gamma,nvar)
  call compute_flux(uright,fright,gamma,nvar)
  ! Compute Godunox flux
  fgdnv=0.5*(fright+fleft)-0.5*cmax*(uright-uleft)
end subroutine riemann_llf
!==============================================
subroutine riemann_hllc(ul,ur,fgdnv,gamma,nvar)
  implicit none
  integer::nvar
  real(kind=8)::gamma
  real(kind=8),dimension(1:nvar)::ul,ur
  real(kind=8),dimension(1:nvar)::fgdnv

  real(kind=8)::cl,cr,dl,dr,sl,sr
  real(kind=8),dimension(1:nvar)::wl,wr,wstar,wstarl,wstarr
  real(kind=8),dimension(1:nvar)::ustarl,ustarr,wgdnv,ugdnv
  ! Compute primitive variables
  call compute_primitive(ul,wl,gamma,nvar)
  call compute_primitive(ur,wr,gamma,nvar)
  ! Compute sound speed
  cl=sqrt(gamma*max(wl(3),1d-10)/max(wl(1),1d-10))
  cr=sqrt(gamma*max(wr(3),1d-10)/max(wr(1),1d-10))
  ! Compute HLL wave speed
  SL=min(wl(2),wr(2))-max(cl,cr)
  SR=max(wl(2),wr(2))+max(cl,cr)
  ! Compute Lagrangian sound speed
  DL=wl(1)*(wl(2)-SL)
  DR=wr(1)*(SR-wr(2))
  ! Compute acoustic star state
  wstar(2)=(DR*wr(2)+DL*wl(2)+(wl(3)-wr(3)))/(DL+DR)
  wstar(3)=(DR*wl(3)+DL*wr(3)+DL*DR*(wl(2)-wr(2)))/(DL+DR)
  ! Left star region states
  wstarl(1)=wl(1)*(SL-wl(2))/(SL-wstar(2))
  ustarl(3)=((SL-wl(2))*ul(3)-wl(3)*wl(2)+wstar(3)*wstar(2))/(SL-wstar(2))
  ! Left star region states
  wstarr(1)=wr(1)*(SR-wr(2))/(SR-wstar(2))
  ustarr(3)=((SR-wr(2))*ur(3)-wr(3)*wr(2)+wstar(3)*wstar(2))/(SR-wstar(2))
  ! Sample the solution at x/t=0
  if(SL>0.0)then
     wgdnv(1)=wl(1)
     wgdnv(2)=wl(2)
     wgdnv(3)=wl(3)
     ugdnv(3)=ul(3)
  else if(wstar(2)>0.0)then
     wgdnv(1)=wstarl(1)
     wgdnv(2)=wstar(2)
     wgdnv(3)=wstar(3)
     ugdnv(3)=ustarl(3)
  else if(SR>0.0)then
     wgdnv(1)=wstarr(1)
     wgdnv(2)=wstar(2)
     wgdnv(3)=wstar(3)
     ugdnv(3)=ustarr(3)
  else
     wgdnv(1)=wr(1)
     wgdnv(2)=wr(2)
     wgdnv(3)=wr(3)
     ugdnv(3)=ur(3)
  end if
  fgdnv(1)=wgdnv(1)*wgdnv(2)
  fgdnv(2)=wgdnv(1)*wgdnv(2)*wgdnv(2)+wgdnv(3)
  fgdnv(3)=wgdnv(2)*(ugdnv(3)+wgdnv(3))
end subroutine riemann_hllc
!==============================================

subroutine add_viscosity(dudt,u)
    use dg_commons
    implicit none
    real(kind=8),dimension(1:nvar,1:n,1:nx)::u,dudt
    real(kind=8),dimension(1:nvar,1:n,1:nx)::viscous
    real(kind=8),dimension(1:nvar,1:n,1:nx)::flux_vo
    real(kind=8)::visc

    real(kind=8),dimension(1:nvar,1:nquad)::flux_quad
    real(kind=8),dimension(1:nvar,1:n,1:nx)::flux_vol
    real(kind=8),dimension(1:nvar,1:nx)::u_left,u_right
    real(kind=8),dimension(1:nvar,1:nx)::flux_left,flux_right
    real(kind=8),dimension(1:nvar)::flux_riemann,u_tmp
    real(kind=8),dimension(1:nvar,1:nx+1)::flux_face

    integer::icell,i,j,iface,ileft,iright,ivar
    real(kind=8)::legendre,legendre_prime
    real(kind=8)::chsi_left=-1,chsi_right=+1
    real(kind=8)::dx,x_quad
    real(kind=8)::cmax,oneoverdx,c_left,c_right

    dx = 1/dble(n)
    oneoverdx = 1./dx
    visc = 0.01
    ! evaluate viscous flux volume integral
      ! Loop over cells
    do icell=1,nx
         !==================================
         ! Compute flux at quadrature points
         !==================================
         ! Loop over quadrature points
         do j=1,nquad
            flux_quad(1:nvar,j)=0.0
            ! Loop over modes
            do i=1,n
               flux_quad(1:nvar,j)=flux_quad(1:nvar,j)+u(1:nvar,i,icell)*legendre_prime(chsi_quad(j),i-1)
            end do
         end do

         !================================
         ! Compute volume integral DG term
         !================================
         ! Loop over modes
         do i=1,n
            flux_vol(1:nvar,i,icell)=0.0
            ! Loop over quadrature points
            do j=1,nquad
               flux_vol(1:nvar,i,icell)=flux_vol(1:nvar,i,icell)+ &
                    & visc*flux_quad(1:nvar,j)* &
                    & legendre_prime(chsi_quad(j),i-1)* &
                    & w_quad(j)
            end do
         end do

         !==============================
         ! Compute left and right states
         !==============================
         u_left(1:nvar,icell)=0.0
         u_right(1:nvar,icell)=0.0
         ! Loop over modes
         do i=1,n
            u_left(1:nvar,icell)=u_left(1:nvar,icell)+u(1:nvar,i,icell)*legendre(chsi_left,i-1)
            u_right(1:nvar,icell)=u_right(1:nvar,icell)+u(1:nvar,i,icell)*legendre(chsi_right,i-1)
         end do

         flux_left(1:nvar,icell)=0.0
         flux_right(1:nvar,icell)=0.0
         ! Loop over modes
         do i=1,n
            flux_left(1:nvar,icell)=flux_left(1:nvar,icell)+visc*u(1:nvar,i,icell)*legendre_prime(chsi_left,i-1)
            flux_right(1:nvar,icell)=flux_right(1:nvar,icell)+visc*u(1:nvar,i,icell)*legendre_prime(chsi_right,i-1)
         end do

      end do
      ! End loop over cells

      !==========================================
      ! Compute physical flux from Riemann solver
      !==========================================
      ! Loop over faces
      do iface=1,nx+1
         ileft=iface-1
         iright=iface
         if(bc==1)then
            ! Periodic boundary conditions
            if(iface==1)ileft=nx
            if(iface==nx+1)iright=1
         endif
         if(bc==2.or.bc==3)then
            ! Zero gradient boundary conditions
            if(iface==1)ileft=1
            if(iface==nx+1)iright=nx
         endif
         ! Compute physical flux using Riemann solver
         call riemann_llf_viscous(u_right(1:nvar,ileft),u_left(1:nvar,iright), flux_right(1:nvar,ileft),&
                 & flux_left(1:nvar,iright) ,flux_riemann,gamma,nvar)

         flux_face(1:nvar,iface)=flux_riemann
         ! Compute boundary flux for reflexive BC
         if(bc==3.AND.iface==1)then
            u_tmp(1:nvar)=u_left(1:nvar,iright)
            u_tmp(2)=-u_tmp(2)
            call riemann_llf_viscous(u_right(1:nvar,ileft),u_left(1:nvar,iright), flux_right(1:nvar,ileft),&
                    & flux_left(1:nvar,iright) ,flux_riemann,gamma,nvar)
            flux_face(1:nvar,iface)=flux_riemann
         endif
         if(bc==3.AND.iface==nx+1)then
            u_tmp(1:nvar)=u_right(1:nvar,ileft)
            u_tmp(2)=-u_tmp(2)
            call riemann_llf_viscous(u_right(1:nvar,ileft),u_left(1:nvar,iright), flux_right(1:nvar,ileft),&
                    & flux_left(1:nvar,iright) ,flux_riemann,gamma,nvar)
            flux_face(1:nvar,iface)=flux_riemann
         endif
      end do

    ! Loop over cells

    !print*,dudt(1,:,:)
    !pause

    print*,maxval(dudt(1,:,:))
    print*,minval(dudt(1,:,:))
    do icell=1,nx
       ! Loop over modes
       do i=1,n
          dudt(1:nvar,i,icell)= dudt(1:nvar,i,icell) -oneoverdx*(flux_vol(1:nvar,i,icell) &
               & +(flux_face(1:nvar,icell+1)*legendre(chsi_right,i-1) &
               & -flux_face(1:nvar,icell)*legendre(chsi_left,i-1)))
          !print*,'update',oneoverdx*(flux_vol(1:nvar,i,icell) &
          !     & -(flux_face(1:nvar,icell+1)*legendre(chsi_right,i-1) &
          !     & -flux_face(1:nvar,icell)*legendre(chsi_left,i-1)))
       end do
    end do
    !print*,'flux',maxval(flux_vol(1,:,:))
    !print*,minval(flux_vol(1,:,:))

    !print*,'fluxface',maxval(flux_face(1,:))
    !print*,minval(flux_face(1,:))
    print*,maxval(dudt(1,:,:))
    print*,minval(dudt(1,:,:))
    !pause
end subroutine add_viscosity


subroutine riemann_llf_viscous(uleft,uright,fright,fleft,fgdnv,gamma,nvar)
  implicit none
  integer::nvar
  real(kind=8)::gamma
  real(kind=8),dimension(1:nvar)::uleft,uright
  real(kind=8),dimension(1:nvar)::fgdnv
  real(kind=8)::cleft,cright,cmax
  real(kind=8),dimension(1:nvar)::fleft,fright
  ! Maximum wave speed
  call compute_speed(uleft,cleft,gamma,nvar)
  call compute_speed(uright,cright,gamma,nvar)
  cmax=max(cleft,cright)
  ! Compute flux at left and right points
  ! Compute Godunox flux
  fgdnv=0.5*(fright+fleft)-0.5*cmax*(uright-uleft)
end subroutine riemann_llf_viscous
