CC=gcc
MPICC=mpicc
CFLAGS=-lm -fopenmp

all: seq_main openmp_main hybrid_main

seq_main: seq_main.c
	$(CC) $(CFLAGS) -o seq_main seq_main.c

openmp_main: openmp_main.c
	$(CC) $(CFLAGS) -o openmp_main openmp_main.c

hybrid_main: hybrid_main.c
	$(MPICC) $(CFLAGS) -o hybrid_main hybrid_main.c

clean:
	rm -f seq_main openmp_main hybrid_main
