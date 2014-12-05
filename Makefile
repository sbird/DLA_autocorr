ifeq ($(CC),cc)
  GCCV:=$(shell gcc --version | head -1)
  ifeq (4.8,$(findstring 4.8,${GCCV}))
  	CC = gcc
  	CXX = g++
  endif
  ifeq ($(CC),cc)
    ICC:=$(shell which icc --tty-only 2>&1)
    #Can we find icc?
    ifeq (/icc,$(findstring /icc,${ICC}))
       CC = icc -vec_report0
       CXX = icpc
    else
       GCC:=$(shell which gcc --tty-only 2>&1)
       #Can we find gcc?
       ifeq (/gcc,$(findstring /gcc,${GCC}))
          CC = gcc
          CXX = g++
       endif
    endif
  endif
endif

#Are we using gcc or icc?
ifeq (icc,$(findstring icc,${CC}))
  CFLAGS +=-O2 -g -c -w1 -openmp -fpic -DNO_KAHAN
  LINK +=${CXX} -openmp
else
  #-ffast-math speeds up by a factor of four!
  CFLAGS +=-O2 -march=native -c -Wall -fopenmp -fPIC -DNO_KAHAN -ffast-math
  LINK +=${CXX} -openmp $(PRO)
  LFLAGS += -lm -lgomp
endif
#Python include path
PYINC=$(shell pkg-config --cflags python2)

.PHONY: all clean

all: moments total _autocorr_priv.so

clean: 
	rm *.o moments total

%.o: %.c
	$(CC) $(CFLAGS) -std=gnu99 -c $^

%.o: %.cpp moments.h
	$(CXX) $(CFLAGS) -c $^

moments: main.o read_hdf_snapshot.o SPH_fieldize.o handle_field.o powerspectrum.o moments.h
	$(LINK) $(LFLAGS) -lfftw3f -lfftw3f_threads -lhdf5 -lhdf5_cpp $^ -o $@

total: main_total.o read_hdf_snapshot.o SPH_fieldize.o CiC_fieldize.o handle_field.o powerspectrum.o moments.h
	$(LINK) $(LFLAGS) -lfftw3f -lfftw3f_threads -lhdf5 -lhdf5_cpp $^ -o $@

py_autocorr.o: py_autocorr.cpp
	$(CXX) $(CFLAGS) -fno-strict-aliasing -DNDEBUG $(PYINC) -c $< -o $@

_autocorr_priv.so: py_autocorr.o
	$(LINK) $(LFLAGS) -shared $^ -o $@ -lgsl -lgslcblas 

