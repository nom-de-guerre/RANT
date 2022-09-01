OPTIONS= $(OUTSIDE)
CC=g++ -std=c++17
DEBUG=-O3 
INCLUDES=-I../NNm -I../common
CFLAGS=-Wall -I. $(OPTIONS) $(DEBUG) $(INCLUDES)
HDEPS = $(wildcard *.h) $(wildcard ../common/*h) $(wildcard ../NNm/*)
DEPS = Makefile $(HDEPS)

diabetes: diabetes.cc $(DEPS) $(HDEPS)
	$(CC) -o $@ $(CFLAGS) diabetes.cc

enzymes: enzymes.cc $(DEPS) $(HDEPS)
	$(CC) -o $@ $(CFLAGS) enzymes.cc

water: water.cc $(DEPS) $(HDEPS)
	$(CC) -o $@ $(CFLAGS) water.cc

MNIST: MNIST.cc $(DEPS) $(HDEPS)
	$(CC) -o $@ $(CFLAGS) MNIST.cc

iris: iris.cc $(DEPS) $(HDEPS)
	$(CC) -o $@ $(CFLAGS) iris.cc

classify: classify.cc $(DEPS) $(HDEPS)
	$(CC) -o $@ $(CFLAGS) classify.cc

sine: sine.cc $(DEPS) $(HDEPS)
	$(CC) -o $@ $(CFLAGS) sine.cc

sine_verify: sine_verify.cc $(DEPS) $(HDEPS)
	$(CC) -o $@ $(CFLAGS) sine_verify.cc

all: sine classify iris sine_verify MNIST water enzymes diabetes

clean:
	rm classify
	rm sine
	rm iris
	rm sine_verify
	rm MNIST
	rm water
	rm enzymes
	rm diabetes
