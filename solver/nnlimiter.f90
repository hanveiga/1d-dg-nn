subroutine detect_shock(uc_modes, up_modes, um_modes,label)
  use ml_parameters
  use dg_commons
  ! performs the shock detection and returns a label
  REAL(KIND=8), DIMENSION(1:nvar, 1:n):: uc_modes, up_modes, um_modes!, reconstruct_cell
  REAL(KIND=8), DIMENSION(1:n_input,1:nvar):: feature_vector
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

  REAL(KIND=8), DIMENSION(1:n_input, 1:nvar):: feature_vector
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
    
    feature_vector(:,ivar) = (/ h, u_c0, u_m0,u_p0, du0, du_m0, du_p0, u_f_p0, u_f_m0, u_m_f_p0, u_p_f_m0 /)
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
    real(kind=8)::read_temp

    !OPEN(1,file=trim(model_folder)//'b5.txt')
    !DO i=1, SIZE(b5)
    !   READ(1,*) b5(i)
    !ENDDO
    !CLOSE(1)



    ! Reading stuff in a more natural manner !this should be the transpose of the stuff above

    OPEN(1,file=trim(model_folder)//'w0.txt')
    write(*,*) 'opened'
    DO i=1,SIZE(coeff01,2)
       READ(1,*) (coeff01(j,i),j=1,SIZE(coeff01,1))
       write(*,*) 'reading'
    ENDDO
    CLOSE(1)

    OPEN(1,file=trim(model_folder)//'w1.txt')
    DO i=1,SIZE(coeff11,2)
       READ(1,*) (coeff11(j,i),j=1,SIZE(coeff11,1))
    ENDDO
    CLOSE(1)

    OPEN(1,file=trim(model_folder)//'w2.txt')
    DO i=1,SIZE(coeff21,2)
       READ(1,*) (coeff21(j,i),j=1,SIZE(coeff21,1))
    ENDDO
    CLOSE(1)

    OPEN(1,file=trim(model_folder)//'w3.txt')
    DO i=1,SIZE(coeff31,2)
       READ(1,*) (coeff31(j,i),j=1,SIZE(coeff31,1))
    ENDDO
    CLOSE(1)

    OPEN(1,file=trim(model_folder)//'w4.txt')
    DO i=1,SIZE(coeff41,2)
       READ(1,*) (coeff41(j,i),j=1,SIZE(coeff41,1))
    ENDDO
    CLOSE(1)

    !OPEN(1,file=trim(model_folder)//'w5.txt')
    !DO i=1,SIZE(coeff51,2)
    !   READ(1,*) (coeff51(j,i),j=1,SIZE(coeff51,1))
    !ENDDO
    !CLOSE(1)

    OPEN(1,file=trim(model_folder)//'b0.txt')
    DO i=1, SIZE(b01,1)
       READ(1,*) read_temp
       b01(i,:) = read_temp
    ENDDO
    CLOSE(1)

    OPEN(1,file=trim(model_folder)//'b1.txt')
    DO i=1, SIZE(b11,1)
       READ(1,*) read_temp
       b11(i,:) = read_temp
    ENDDO
    CLOSE(1)

    OPEN(1,file=trim(model_folder)//'b2.txt')
    DO i=1, SIZE(b21,1)
       READ(1,*) read_temp
       b21(i,:) = read_temp
    ENDDO
    CLOSE(1)

    OPEN(1,file=trim(model_folder)//'b3.txt')
    DO i=1, SIZE(b31,1)
       READ(1,*) read_temp
       b31(i,:) = read_temp
    ENDDO
    CLOSE(1)

    OPEN(1,file=trim(model_folder)//'b4.txt')
    DO i=1, SIZE(b41,1)
       READ(1,*) read_temp
       b41(i,:) = read_temp
    ENDDO
    CLOSE(1)

    !OPEN(1,file=trim(model_folder)//'b5.txt')
    !DO i=1, SIZE(b51,1)
    !   READ(1,*) read_temp
    !   b51(i,:) = read_temp
    !ENDDO
    !CLOSE(1)

    !print*, maxval(coeff0 - transpose(coeff01)), minval(coeff0 - transpose(coeff01))
    !print*, maxval(coeff1 - transpose(coeff11)), minval(coeff1 - transpose(coeff11))
    !print*, maxval(coeff2 - transpose(coeff21)), minval(coeff2 - transpose(coeff21))
    !print*, maxval(coeff3 - transpose(coeff31)), minval(coeff3 - transpose(coeff31))
    !print*, maxval(coeff4 - transpose(coeff41)), minval(coeff4 - transpose(coeff41))
    !print*, maxval(coeff5 - transpose(coeff51)), minval(coeff5 - transpose(coeff51))

    !pause
    
    !print*, coeff1(1,:)
    !print*,coeff11(:,1)
    ! probably ok

end subroutine

subroutine ml_shock_detector(x,label)
  use ml_parameters
  use dg_commons
  IMPLICIT NONE
  REAL(KIND=8), DIMENSION(1:n_input,1:nvar):: x
  !REAL(KIND=8), DIMENSION(1:n_input,1:nvar,1:n_input):: x_transposed
  real(kind=8)::alpha=1.0,beta=0.0

  INTEGER,dimension(1:nvar)::label

  integer:: i,j, var

  !x_transposed = transpose(x)

  !do var = 1,nvar 
  !var=1
    !y1=MAX(MATMUL(x(var, :) ,coeff0)+b0,0.)
    !y2=MAX(MATMUL(y1,coeff1)+b1,0.)
    !y3=MAX(MATMUL(y2,coeff2)+b2,0.)
    !y4=MAX(MATMUL(y3,coeff3)+b3,0.)
    !y5=MAX(MATMUL(y4,coeff4)+b4,0.)
    
    !print*, y1
    !y1 = MAX(MATMUL(coeff01,x_transposed(:,var))+b01(:,var),0.)
    !print*, y1

    !pause
    ! getting rid of matmul
    ! output layer: softmax
    !y6 = MATMUL(y5,coeff5)+b5
    !y6 = exp(y6-maxval(y6))/sum(exp(y6-maxval(y6)))
    !write(*,*) y6

    !CALL DGEMM('N','N',1,layer1,n_input,alpha,x(var,:),1,coeff0,n_input,beta,y1,1)
    !y1 = max(y1+b0,0.)
    !CALL DGEMM('N','N',1,layer2,layer1,alpha,y1,1,coeff1,layer1,beta,y2,1)
    !y2 = max(y2+b1,0.)
    !CALL DGEMM('N','N',1,layer3,layer2,alpha,y2,1,coeff2,layer2,beta,y3,1)
    !y3 = max(y3+b2,0.)
    !CALL DGEMM('N','N',1,layer4,layer3,alpha,y3,1,coeff3,layer3,beta,y4,1)
    !y4 = max(y4+b3,0.)
    !CALL DGEMM('N','N',1,layer5,layer4,alpha,y4,1,coeff4,layer4,beta,y5,1)
    !y5 = max(y5+b4,0.)

    !CALL DGEMM('N','N',1,nclass,layer5,alpha,y5,1,coeff5,layer5,beta,y6,1)
    !y6 = y6 + b5
    !y6 = exp(y6-maxval(y6))/sum(exp(y6-maxval(y6)))


    ! optimized

    ! gemm('N', 'N', m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc)

    CALL DGEMM('N','N',layer1,nvar,n_input,alpha,coeff01,layer1,x,n_input,beta,y11,layer1)
    y11 = max(y11+b01,0.)
    !print*, maxval(abs(y1 - y11(:,var)))
    CALL DGEMM('N','N',layer2,nvar,layer1,alpha,coeff11,layer2,y11,layer1,beta,y21,layer2)
    y21 = max(y21+b11,0.)
    !print*, maxval(abs(y2 - y21(:,var)))

    CALL DGEMM('N','N',layer3,nvar,layer2,alpha,coeff21,layer3,y21,layer2,beta,y31,layer3)
    y31 = max(y31+b21,0.)
    !print*, maxval(abs(y3 - y31(:,var)))

    CALL DGEMM('N','N',layer4,nvar,layer3,alpha,coeff31,layer4,y31,layer3,beta,y41,layer4)
    y41 = max(y41+b31,0.)
    !print*, maxval(abs(y4 - y41(:,var)))

    !CALL DGEMM('N','N',layer5,nvar,layer4,alpha,coeff41,layer5,y41,layer4,beta,y51,layer5)
    !y51 = max(y51+b41,0.)
    !print*, maxval( abs(y5 - y51(:,var)))

    !pause

    CALL DGEMM('N','N',nclass,nvar,layer4,alpha,coeff41,nclass,y41,layer4,beta,y51,nclass)
    y61 = y51 + b51
    


    !write(*,*) y6
    !pause

    do var = 1, nvar
      y61(:,var) = exp(y61(:,var)-maxval(y61(:,var)))/sum(exp(y61(:,var)-maxval(y61(:,var))))
      !print*,y61(:,var)
      if (y61(2,var) >= 0.5) then
        label(var) = 1
      else
        label(var) = 0
      end if
    end do
    !if ((y61(2,var)>=0.5).and.( label(var) == 0)) then
    !  print*, y61(:,var), y6
    !  print*, 'thisis a big problem' !!! BUG
    !  pause
    !end if

  !end do
END subroutine ml_shock_detector
