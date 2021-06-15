OPTIONS= $(OUTSIDE)
CC=g++
DEBUG=-O3 
INCLUDES=-I../NNm -I../common
CFLAGS=-Wall -I. $(OPTIONS) $(DEBUG) $(INCLUDES)
HDEPS = $(wildcard *.h) $(wildcard ../common/*h) $(wildcard ../NNm/*)
DEPS = Makefile $(HDEPS)

classify: classify.cc $(DEPS) $(HDEPS)
	$(CC) -o $@ $(CFLAGS) classify.cc

sine: sine.cc $(DEPS) $(HDEPS)
	$(CC) -o $@ $(CFLAGS) sine.cc

all: sine classify

clean:
	rm classify
	rm sine
