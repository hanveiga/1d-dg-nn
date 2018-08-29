subroutine detect_shock(uc_modes, up_modes, um_modes,label)
  use ml_parameters
  use dg_commons
  ! performs the shock detection and returns a label
  REAL(KIND=8), DIMENSION(1:nvar, 1:n):: uc_modes, up_modes, um_modes!, reconstruct_cell
  REAL(KIND=8), DIMENSION(1:nvar,1:n_input):: feature_vector
  integer,dimension(1:nvar)::label
  feature_vector = 0.0
  call generate_features(uc_modes,up_modes,um_modes,feature_vector)
  call ml_shock_detector(feature_vector, label)

end subroutine

subroutine generate_features_old(uc_modes, up_modes, um_modes, feature_vector)
  use ml_parameters
  use dg_commons
  IMPLICIT NONE

  REAL(KIND=8), DIMENSION(1:nvar, 1:n_input):: feature_vector
  REAL(KIND=8), DIMENSION(1:nvar, 1:n):: uc_modes, up_modes, um_modes!, reconstruct_cell
  REAL(KIND=8), dimension(1:nvar):: u_c, u_m, u_p, du_m, du_p, u_f_p, u_f_m, max_u, du!, reconstruct_cell
  REAL(KIND=8):: h, pos
  integer:: ivar

  h = boxlen/dble(nx)
  pos = 0.0
  call reconstruct_cell(uc_modes,pos, u_c)
  pos = 0.0
  call reconstruct_cell(um_modes,pos, u_m)
  pos = 0.0
  call reconstruct_cell(up_modes,pos, u_p)

  do ivar = 1,nvar
    max_u(ivar) = maxval( abs((/ u_c(ivar), u_m(ivar), u_p(ivar) /)))
  end do

  pos = 1.0
  call reconstruct_cell(uc_modes,pos, u_f_p)
  pos = -1.0
  call reconstruct_cell(uc_modes,pos, u_f_m)
  !write(*,*) 'generated features'
  !u_m = reconstruct_cell(up_modes,n,0.0,nvar)
  !u_p = reconstruct_cell(um_modes,n,0.0,nvar)
  !do ivar = 1,nvar
  !  u_m(ivar) = u_m(ivar)!/max_u(ivar) 
  !  u_c(ivar) = u_c(ivar)!/max_u(ivar)
  !  u_p = u_p(ivar)!/max_u(ivar)
  !  u_f_p = u_f_p(ivar)!/max_u(ivar)
  !  u_f_m = u_f_m(ivar)!/max_u(ivar)
  !end do

  du_m = (u_c - u_m)/h
  du_p = (u_p - u_c)/h
  du = (u_p - u_m)/(2*h)
  !u_f_p = reconstruct_cell(uc_modes,n,1.0,nvar)
  !u_f_m = reconstruct_cell(uc_modes,n,-1.0,nvar)

  !feature_vector(:) = (/ h, u_c, u_m, u_p, du_m, &
  !            &du_p, u_f_p, u_f_m/)

  !dataset.append([xv,h,u_m, u_c, u_p, du, du_m, du_p, u_f_p, u_f_m, u_max, label,label_hio])
        
  do ivar = 1,nvar
  !feature_vector(ivar,:) = (/ h, u_m(ivar), u_c(ivar), u_p(ivar), du(ivar), du_m(ivar), &
  !            &du_p(ivar), u_f_p(ivar), u_f_m(ivar) /)

  ! GOOD
  !feature_vector(ivar,:) = (/ h, u_c(ivar),u_m(ivar), u_p(ivar), du_m(ivar), &
  !            &du_p(ivar), u_f_p(ivar), u_f_m(ivar) /)
  
  !feature_vector(ivar,:) = (/ h, u_m(ivar),u_c(ivar), u_p(ivar), du(ivar), du_m(ivar), &
  !            &du_p(ivar), u_f_p(ivar), u_f_m(ivar) /)
  end do

end subroutine

subroutine generate_features(uc_modes, up_modes, um_modes, feature_vector)
  use ml_parameters
  use dg_commons
  IMPLICIT NONE

  REAL(KIND=8), DIMENSION(1:nvar, 1:n_input):: feature_vector
  REAL(KIND=8), DIMENSION(1:nvar, 1:n):: uc_modes, up_modes, um_modes!, reconstruct_cell
  REAL(KIND=8), dimension(1:nvar):: u_c, u_m, u_p, du_m, du_p, u_f_p, u_f_m, du!, reconstruct_cell
  REAL(KIND=8), dimension(1:nvar):: u_m_f_p, u_p_f_m
  REAL(KIND=8):: h, pos
  REAL(KIND=8):: max_u, min_u, u_c0, u_m0, u_p0, du_m0, du_p0, u_f_p0, u_f_m0, du0, u_m_f_p0, u_p_f_m0
  
  integer:: ivar

  !normal = [xv,h, u_c, u_m,u_p, du, du_m, du_p, u_f_p, u_f_m, u_m_f_p, u_p_f_m, u_max, u_min, label_hio]

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

  pos = 1.0
  call reconstruct_cell(um_modes,pos, u_m_f_p)
  pos = -1.0
  call reconstruct_cell(up_modes,pos, u_p_f_m)

  du_m = (u_c - u_m)/h
  du_p = (u_p - u_c)/h
  du = (u_p - u_m)/(2*h)

  do ivar = 1,nvar
    max_u = 1.0 !maxval( (/ u_c(ivar), u_m(ivar), u_p(ivar) /))
    min_u = 0.0 !minval( (/ u_c(ivar), u_m(ivar), u_p(ivar) /))
    !write(*,*) max_u
    ! normalize features
    if ((max_u.eq.min_u).and.(abs(max_u)>0)) then !.and.(abs(max_u)>0)) then
      u_c0 = u_c(ivar)/max_u
      u_m0 = u_m(ivar)/max_u
      u_p0 = u_p(ivar)/max_u
      du0 = du(ivar)/max_u
      du_m0 = du_m(ivar)/max_u 
      du_p0 = du_p(ivar)/max_u !entry[7]/float(maxval)
      u_f_p0 = u_f_p(ivar)/max_u !entry[8]/float(maxval)
      u_f_m0 = u_f_m(ivar)/max_u !entry[9]/float(maxval)
      u_m_f_p0 = u_m_f_p(ivar)/max_u ! entry[10]/float(maxval)
      u_p_f_m0 = u_p_f_m(ivar)/max_u !entry[11]/float(maxval)
    else if ((max_u == 0).and.(min_u==max_u)) then
      max_u = 1.0
      u_c0 = u_c(ivar)/max_u
      u_m0 = u_m(ivar)/max_u
      u_p0 = u_p(ivar)/max_u
      du0 = du(ivar)/max_u
      du_m0 = du_m(ivar)/max_u 
      du_p0 = du_p(ivar)/max_u !entry[7]/float(maxval)
      u_f_p0 = u_f_p(ivar)/max_u !entry[8]/float(maxval)
      u_f_m0 = u_f_m(ivar)/max_u !entry[9]/float(maxval)
      u_m_f_p0 = u_m_f_p(ivar)/max_u ! entry[10]/float(maxval)
      u_p_f_m0 = u_p_f_m(ivar)/max_u !entry[11]/float(maxval)
    else
      u_c0 = (u_c(ivar)-min_u)/(max_u-min_u)
      u_m0 = (u_m(ivar)-min_u)/(max_u-min_u)
      u_p0 = (u_p(ivar)-min_u)/(max_u-min_u)
      du0 = (du(ivar)-min_u)/(max_u-min_u)
      du_m0 = (du_m(ivar)-min_u)/(max_u-min_u)
      du_p0 = (du_p(ivar)-min_u)/(max_u-min_u)
      u_f_p0 = (u_f_p(ivar)-min_u)/(max_u-min_u)
      u_f_m0 = (u_f_m(ivar)-min_u)/(max_u-min_u)
      u_m_f_p0 = (u_m_f_p(ivar)-min_u)/(max_u-min_u)
      u_p_f_m0 = (u_p_f_m(ivar)-min_u)/(max_u-min_u)
    end if
    
    feature_vector(ivar,:) = (/ h, u_c0, u_m0,u_p0, du0, du_m0, du_p0, u_f_p0, u_f_m0, u_m_f_p0, u_p_f_m0 /)
  end do

end subroutine


subroutine reconstruct_cell(modes, pos, values)  !result(values)
  use dg_commons
  implicit none
  REAL(KIND=8), DIMENSION(1:nvar,1:n):: modes
  REAL(KIND=8):: legendre
  REAL(KIND=8),dimension(1:nvar):: values
  real(kind=8)::pos
  integer::i,var
  
  values = 0.
  
  do var = 1, nvar
    do i = 1, n
        values(var) = values(var) + modes(var,i)*legendre(pos,i-1)
    end do
  end do

end subroutine

subroutine load_nn()
    use ml_parameters
    use dg_commons
    implicit none
    integer::i,j

    OPEN(1,file=trim(model_folder)//'w0.txt')
    write(*,*) 'opened'
    DO i=1,SIZE(coeff0,1)
       READ(1,*) (coeff0(i,j),j=1,SIZE(coeff0,2))
       write(*,*) 'reading'
    ENDDO
    CLOSE(1)

    OPEN(1,file=trim(model_folder)//'w1.txt')
    DO i=1,SIZE(coeff1,1)
       READ(1,*) (coeff1(i,j),j=1,SIZE(coeff1,2))
    ENDDO
    CLOSE(1)

    OPEN(1,file=trim(model_folder)//'w2.txt')
    DO i=1,SIZE(coeff2,1)
       READ(1,*) (coeff2(i,j),j=1,SIZE(coeff2,2))
    ENDDO
    CLOSE(1)

    OPEN(1,file=trim(model_folder)//'w3.txt')
    DO i=1,SIZE(coeff3,1)
       READ(1,*) (coeff3(i,j),j=1,SIZE(coeff3,2))
    ENDDO
    CLOSE(1)

    OPEN(1,file=trim(model_folder)//'w4.txt')
    DO i=1,SIZE(coeff4,1)
       READ(1,*) (coeff4(i,j),j=1,SIZE(coeff4,2))
    ENDDO
    CLOSE(1)

    OPEN(1,file=trim(model_folder)//'w5.txt')
    DO i=1,SIZE(coeff5,1)
       READ(1,*) (coeff5(i,j),j=1,SIZE(coeff5,2))
    ENDDO
    CLOSE(1)

    OPEN(1,file=trim(model_folder)//'b0.txt')
    DO i=1, SIZE(b0)
       READ(1,*) b0(i)
    ENDDO
    CLOSE(1)

    OPEN(1,file=trim(model_folder)//'b1.txt')
    DO i=1, SIZE(b1)
       READ(1,*) b1(i)
    ENDDO
    CLOSE(1)

    OPEN(1,file=trim(model_folder)//'b2.txt')
    DO i=1, SIZE(b2)
       READ(1,*) b2(i)
    ENDDO
    CLOSE(1)

    OPEN(1,file=trim(model_folder)//'b3.txt')
    DO i=1, SIZE(b3)
       READ(1,*) b3(i)
    ENDDO
    CLOSE(1)

    OPEN(1,file=trim(model_folder)//'b4.txt')
    DO i=1, SIZE(b4)
       READ(1,*) b4(i)
    ENDDO
    CLOSE(1)

    OPEN(1,file=trim(model_folder)//'b5.txt')
    DO i=1, SIZE(b5)
       READ(1,*) b5(i)
    ENDDO
    CLOSE(1)
end subroutine

subroutine ml_shock_detector(x,label)
  use ml_parameters
  use dg_commons
  IMPLICIT NONE
  REAL(KIND=8), DIMENSION(1:nvar,1:n_input):: x
  real(kind=8)::alpha=1.0,beta=0.0

  INTEGER,dimension(1:nvar)::label

  integer:: i,j, var

  do var = 1,nvar 
  !var=1
    !y1=MAX(MATMUL(x(var, :) ,coeff0)+b0,0.)
    !y2=MAX(MATMUL(y1,coeff1)+b1,0.)
    !y3=MAX(MATMUL(y2,coeff2)+b2,0.)
    !y4=MAX(MATMUL(y3,coeff3)+b3,0.)
    !y5=MAX(MATMUL(y4,coeff4)+b4,0.)

    ! getting rid of matmul
    ! output layer: softmax
    !y6 = MATMUL(y5,coeff5)+b5
    !y6 = exp(y6-maxval(y6))/sum(exp(y6-maxval(y6)))
    !write(*,*) y6


    CALL DGEMM('N','N',1,layer1,n_input,alpha,x(var,:),1,coeff0,n_input,beta,y1,1)
    y1 = max(y1+b0,0.)
    CALL DGEMM('N','N',1,layer2,layer1,alpha,y1,1,coeff1,layer1,beta,y2,1)
    y2 = max(y2+b1,0.)
    CALL DGEMM('N','N',1,layer3,layer2,alpha,y2,1,coeff2,layer2,beta,y3,1)
    y3 = max(y3+b2,0.)
    CALL DGEMM('N','N',1,layer4,layer3,alpha,y3,1,coeff3,layer3,beta,y4,1)
    y4 = max(y4+b3,0.)
    CALL DGEMM('N','N',1,layer5,layer4,alpha,y4,1,coeff4,layer4,beta,y5,1)
    y5 = max(y5+b4,0.)

    CALL DGEMM('N','N',1,nclass,layer5,alpha,y5,1,coeff5,layer5,beta,y6,1)
    y6 = exp(y6-maxval(y6))/sum(exp(y6-maxval(y6)))
    !write(*,*) y6
    !pause


    if (y6(2) >= 0.5) then
      label(var) = 1
    else
      label(var) = 0
    end if

  end do
END subroutine ml_shock_detector
