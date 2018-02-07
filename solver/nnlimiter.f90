subroutine detect_shock(uc_modes, up_modes, um_modes,label)
  use ml_parameters
  use dg_commons
  ! performs the shock detection and returns a label
  REAL(KIND=8), DIMENSION(1:nvar,1:n):: uc_modes, up_modes, um_modes!, reconstruct_cell
  REAL(KIND=8), DIMENSION(1:nvar,1:n_input):: feature_vector
  integer::label
  feature_vector = 0.0
  call generate_features(uc_modes,up_modes,um_modes,feature_vector)
  call ml_shock_detector(feature_vector, label)

end subroutine

subroutine generate_features(uc_modes, up_modes, um_modes, feature_vector)
  use ml_parameters
  use dg_commons
  IMPLICIT NONE

  REAL(KIND=8), DIMENSION(1:nvar,1:n_input):: feature_vector
  REAL(KIND=8), DIMENSION(1:nvar,1:n):: uc_modes, up_modes, um_modes!, reconstruct_cell
  REAL(KIND=8), DIMENSION(1:nvar):: u_c, u_m, u_p, du_m, du_p, u_f_p, u_f_m!, reconstruct_cell
  REAL(KIND=8):: h, pos
  integer:: var

  h = boxlen/dble(nx)
  pos = 0.0
  call reconstruct_cell(uc_modes,pos, u_c)
  pos = 0.0
  call reconstruct_cell(um_modes,pos, u_m)
  pos = 0.0
  call reconstruct_cell(up_modes,pos, u_p)
  pos = 1.0
  call reconstruct_cell(uc_modes,pos, u_f_p)
  pos = -1.0
  call reconstruct_cell(uc_modes,pos, u_f_m)
  !write(*,*) 'generated features'
  !u_m = reconstruct_cell(up_modes,n,0.0,nvar)
  !u_p = reconstruct_cell(um_modes,n,0.0,nvar)
  !du_m = (u_c - u_m)/h
  !du_p = (u_p - u_c)/h
  !u_f_p = reconstruct_cell(uc_modes,n,1.0,nvar)
  !u_f_m = reconstruct_cell(uc_modes,n,-1.0,nvar)

  do var = 1,nvar
    feature_vector(var,:) = (/ h, u_c(var), u_m(var), u_p(var), du_m(var), &
              &du_p(var), u_f_p(var), u_f_m(var)/)
  end do

end subroutine

subroutine reconstruct_cell(modes, pos, values)  !result(values)
  use dg_commons
  implicit none
  REAL(KIND=8), DIMENSION(1:nvar,1:n):: modes
  REAL(KIND=8):: legendre
  REAL(KIND=8), DIMENSION(1:nvar):: values
  real(kind=8)::pos
  integer::i,var
  !write(*,*) 'reconstructing cell'
  values = 0.
  !write(*,*) values
  !write(*,*) modes
  do var = 1, nvar
    do i = 1, n
        !write(*,*) 'inside loop'
        !write(*,*) i, var
        !write(*,*) legendre(dble(0.0),i-1)
        values(var) = values(var) + modes(nvar,i)*legendre(pos,i-1)
    end do
  end do

end subroutine

subroutine load_nn()
    use ml_parameters
    use dg_commons
    implicit none
    integer::i,j

    OPEN(1,file='w0.txt')
    DO i=1,SIZE(coeff0,1)
       READ(1,*) (coeff0(i,j),j=1,SIZE(coeff0,2))
    ENDDO
    CLOSE(1)

    OPEN(1,file='w1.txt')
    DO i=1,SIZE(coeff1,1)
       READ(1,*) (coeff1(i,j),j=1,SIZE(coeff1,2))
    ENDDO
    CLOSE(1)

    OPEN(1,file='w2.txt')
    DO i=1,SIZE(coeff2,1)
       READ(1,*) (coeff2(i,j),j=1,SIZE(coeff2,2))
    ENDDO
    CLOSE(1)

    OPEN(1,file='w3.txt')
    DO i=1,SIZE(coeff3,1)
       READ(1,*) (coeff3(i,j),j=1,SIZE(coeff3,2))
    ENDDO
    CLOSE(1)

    OPEN(1,file='w4.txt')
    DO i=1,SIZE(coeff4,1)
       READ(1,*) (coeff4(i,j),j=1,SIZE(coeff4,2))
    ENDDO
    CLOSE(1)

    OPEN(1,file='b0.txt')
    DO i=1, SIZE(b0)
       READ(1,*) b0(i)
    ENDDO
    CLOSE(1)

    OPEN(1,file='b1.txt')
    DO i=1, SIZE(b1)
       READ(1,*) b1(i)
    ENDDO
    CLOSE(1)

    OPEN(1,file='b2.txt')
    DO i=1, SIZE(b2)
       READ(1,*) b2(i)
    ENDDO
    CLOSE(1)

    OPEN(1,file='b3.txt')
    DO i=1, SIZE(b3)
       READ(1,*) b3(i)
    ENDDO
    CLOSE(1)

    OPEN(1,file='b4.txt')
    DO i=1, SIZE(b4)
       READ(1,*) b4(i)
    ENDDO
    CLOSE(1)

end subroutine

subroutine ml_shock_detector(x,label)
  use ml_parameters
  use dg_commons
  IMPLICIT NONE
  REAL(KIND=8), DIMENSION(1:nvar,1:n_input):: x

  INTEGER::label

  integer:: i,j

  y1=MAX(MATMUL(x(1,:) ,coeff0)+b0,0.)
  y2=MAX(MATMUL(y1,coeff1)+b1,0.)
  y3=MAX(MATMUL(y2,coeff2)+b2,0.)
  y4=MAX(MATMUL(y3,coeff3)+b3,0.)

  ! output layer: softmax
  y5 = MATMUL(y4,coeff4)+b4
  y5 = exp(y5-maxval(y5))/sum(exp(y5-maxval(y5)))

  !PRINT*, 'output'
  !print*, x
  !print*, maxval(y1), minval(y1)
  !print*, maxval(y2), minval(y2)
  !print*, maxval(y3), minval(y3)
  !print*, maxval(y4), minval(y4)
  !print*, maxval(y5), minval(y5)
  !PRINT*, 'y[1]', y5(1)
  !PRINT*, 'y[2]', y5(2)
  if (y5(2) >= 0.5) then
    label = 1
  else
    label = 0
  end if

END subroutine ml_shock_detector
