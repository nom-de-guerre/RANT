OPTIONS= $(OUTSIDE)
CC=g++ -std=c++17
DEBUG=-O3 
ANTROOT=../
INCLUDES=-I$(ANTROOT)/NNm -I$(ANTROOT)/common
CFLAGS=-Wall -I. $(OPTIONS) $(DEBUG) $(INCLUDES)
HDEPS = $(wildcard *.h) $(wildcard ../common/*h) $(wildcard ../NNm/*)

DEPS = Makefile $(HDEPS)

PyRANT.o : PyRANT.cc $(DEPS)
	$(CC) $(CFLAGS) $(DEBUG) -c -fPIC PyRANT.cc

libRANT.so : PyRANT.o
	g++ $(DEBUG) -shared -Wl -o libRANT.so PyRANT.o

all: libRANT.so

clean:
	rm PyRANT.o
	rm libRANT.so
