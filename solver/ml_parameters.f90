module ml_parameters
  !INTEGER, PARAMETER:: layer1=256,layer2=128,layer3=64,layer4=32,layer5=8
  INTEGER, PARAMETER:: layer1=32,layer2=32,layer3=32,layer4=32,layer5=32
  !INTEGER, PARAMETER:: n_input=9
  INTEGER, PARAMETER:: n_input=11
  INTEGER, PARAMETER:: nclass=2
  logical,parameter::normalize=.False.

!256,128,64,8

  REAL(KIND=8), DIMENSION(n_input,layer1):: coeff0
  REAL(KIND=8), DIMENSION(layer1,layer2):: coeff1
  REAL(KIND=8), DIMENSION(layer2,layer3):: coeff2
  REAL(KIND=8), DIMENSION(layer3,layer4):: coeff3
  REAL(KIND=8), DIMENSION(layer4,layer5):: coeff4
  REAL(KIND=8), DIMENSION(layer5,nclass):: coeff5
  REAL(KIND=8), DIMENSION(layer1):: b0,y1
  REAL(KIND=8), DIMENSION(layer2):: b1,y2
  REAL(KIND=8), DIMENSION(layer3):: b2,y3
  REAL(KIND=8), DIMENSION(layer4):: b3,y4
  REAL(KIND=8), DIMENSION(layer5):: b4,y5
  REAL(KIND=8), DIMENSION(nclass):: b5,y6
  character(LEN=*),parameter::model_folder='models/new_l4_w1.0_datadataset_unlimited_0.01_20.4.18_normFalse_nNeurons128/best_'



  INTEGER, PARAMETER:: num_var=3
  REAL(KIND=8), DIMENSION(layer1,n_input):: coeff01
  REAL(KIND=8), DIMENSION(layer2,layer1):: coeff11
  REAL(KIND=8), DIMENSION(layer3,layer2):: coeff21
  REAL(KIND=8), DIMENSION(layer4,layer3):: coeff31
  REAL(KIND=8), DIMENSION(nclass,layer4):: coeff41
  !REAL(KIND=8), DIMENSION(nclass,layer5):: coeff51
  REAL(KIND=8), DIMENSION(layer1,num_var):: b01,y11
  REAL(KIND=8), DIMENSION(layer2,num_var):: b11,y21
  REAL(KIND=8), DIMENSION(layer3,num_var):: b21,y31
  REAL(KIND=8), DIMENSION(layer4,num_var):: b31,y41
  REAL(KIND=8), DIMENSION(nclass,num_var):: b41,y51
  REAL(KIND=8), DIMENSION(nclass,num_var):: b51,y61

  !REAL(KIND=8), DIMENSION(layer1):: y1_test

  !REAL(KIND=8), DIMENSION(layer1):: y1_test2
  !REAL(KIND=8), DIMENSION(layer1,n_input):: coeff0_t
  !character(LEN=*),parameter::model_folder='models/unlimited_data_w5_norm_0.05/'
  !character(LEN=*),parameter::model_folder='models/l3_w5.0_datadataset_unlimited_0.01_20.4.18_normFalse_nNeurons3328/'

end module ml_parameters
