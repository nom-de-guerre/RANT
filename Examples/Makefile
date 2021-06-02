OPTIONS= $(OUTSIDE)
CC=g++
# DEBUG=-O0 -g
# DEBUG=-O3 -mavx
DEBUG=-O3 
CFLAGS=-Wall -I. $(OPTIONS) $(DEBUG)
HDEPS = NNm.h NNm.tcc data.h NeuralM.h
DEPS = Makefile $(HDEPS)

classify: classify.cc $(DEPS) softmax.h
	$(CC) -o $@ $(CFLAGS) classify.cc

sine: sine.cc $(DEPS) regression.h
	$(CC) -o $@ $(CFLAGS) sine.cc

all: sine classify

clean:
	rm classify
	rm sine
