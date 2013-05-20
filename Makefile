#Python include path
PYINC=-I/usr/include/python2.6 -I/usr/include/python2.6

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

#Are we using gcc or icc?
ifeq (icc,$(findstring icc,${CC}))
  CFLAGS +=-O2 -g -c -w1 -openmp -fpic -std=gnu99
  LINK +=${CXX} -openmp
else
  CFLAGS +=-O3 -g -c -Wall -fopenmp -fPIC
  LINK +=${CXX} -openmp $(PRO)
  LFLAGS += -lm -lgomp
endif
.PHONY: all clean

all: moments

clean: 
	rm *.o moments

%.o: %.c
	$(CC) $(CFLAGS) -fPIC -fno-strict-aliasing -DNDEBUG $(PYINC) -c $^ -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) -c $^ -o $@

moments: main.o read_hdf_snapshot.o SPH_fieldize.o handle_field.o
	$(LINK) $(LFLAGS) -lfftw3 -lfftw3_threads -lhdf5 -lhdf5_hl $^ -o $@
