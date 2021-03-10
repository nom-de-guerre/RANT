# DJS_MATH=/Users/dsantry/Desktop/Shuttle/Matrix\ V2/
NN_PATH=NNm
OPTIONS= $(OUTSIDE)
CC=g++
DEBUG=-O3 -mavx
#DEBUG=-O0 -g
CFLAGS=-Wall -I. -I$(NN_PATH) $(OPTIONS) $(DEBUG)
HDEPS = $(NN_PATH)/*.h $(NN_PATH)/NNm.tcc
DEPS = *.h Makefile $(HDEPS)

LeNet5: LeNet5.cc $(DEPS)
	$(CC) -o $@ $(CFLAGS) LeNet5.cc

digits: digits.cc $(DEPS)
	$(CC) -o $@ $(CFLAGS) digits.cc

all: LeNet5 digits

clean:
	rm digits
	rm LeNet5
