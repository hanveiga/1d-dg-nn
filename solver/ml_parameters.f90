module ml_parameters
  INTEGER, PARAMETER:: layer1=256,layer2=128,layer3=64,layer4=32
  INTEGER, PARAMETER:: n_input=8
  INTEGER, PARAMETER:: nclass=2

  REAL(KIND=8), DIMENSION(n_input,layer1):: coeff0
  REAL(KIND=8), DIMENSION(layer1,layer2):: coeff1
  REAL(KIND=8), DIMENSION(layer2,layer3):: coeff2
  REAL(KIND=8), DIMENSION(layer3,layer4):: coeff3
  REAL(KIND=8), DIMENSION(layer4,nclass):: coeff4
  REAL(KIND=8), DIMENSION(layer1):: b0,y1
  REAL(KIND=8), DIMENSION(layer2):: b1,y2
  REAL(KIND=8), DIMENSION(layer3):: b2,y3
  REAL(KIND=8), DIMENSION(layer4):: b3,y4
  REAL(KIND=8), DIMENSION(nclass):: b4,y5

end module ml_parameters
