F90=gfortran
OPT = -O3
MODEL = dg

execname: $(MODEL).o legendre.o dg_commons.o ml_parameters.o nnlimiter.o
	    $(F90) -o dg $(MODEL).o legendre.o dg_commons.o ml_parameters.o nnlimiter.o

ml_parameters.o: ml_parameters.f90
	    $(F90) -c ml_parameters.f90

nnlimiter.o: nnlimiter.f90
	    $(F90) -c nnlimiter.f90

dg_commons.o: dg_commons.f90
	    $(F90) -c dg_commons.f90

legendre.o: legendre.f90
	    $(F90) -c legendre.f90

$(MODEL).o: dg_commons.o legendre.o $(MODEL).f90
	    $(F90) -c $(MODEL).f90

clean:
		rm *.mod
		rm *.o
