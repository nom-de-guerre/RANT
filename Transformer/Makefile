OPTIONS= $(OUTSIDE)
CC=g++ -std=c++17
DEBUG=-O3 
INCLUDES=-I../NNm -I../common
CFLAGS=-Wall -I. $(OPTIONS) $(DEBUG) $(INCLUDES)
# HDEPS = $(wildcard *.h) $(wildcard ../common/*h) $(wildcard ../NNm/*)
HDEPS = $(wildcard *.h)
DEPS = Makefile $(HDEPS)

causal: causal.cc $(DEPS) $(HDEPS)
	$(CC) -o $@ $(CFLAGS) causal.cc

masked: masked.cc $(DEPS) $(HDEPS)
	$(CC) -o $@ $(CFLAGS) masked.cc

all: causal masked

clean:
	rm causal
	rm masked
