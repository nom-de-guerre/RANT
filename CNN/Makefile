NN_INCLUDES = -I../NNm -I../common
OPTIONS= $(OUTSIDE)
CC=g++ -std=c++17
DEBUG=-O3 
#DEBUG=-O0 -g
CFLAGS=-Wall -I. $(NN_INCLUDES) $(OPTIONS) $(DEBUG)
NN_DEPS = $(wildcard *.h) $(wildcard ../common/*h) $(wildcard ../NNm/*)

DEPS = Makefile $(HDEPS) $(NN_DEPS)

LeNet5: LeNet5.cc $(DEPS)
	$(CC) -o $@ $(CFLAGS) LeNet5.cc

digits: digits.cc $(DEPS)
	$(CC) -o $@ $(CFLAGS) digits.cc

all: LeNet5 digits

clean:
	rm digits
	rm LeNet5
