NN_INCLUDES = -I../NNm -I../common -I../CNN
OPTIONS= $(OUTSIDE)
CC=g++
DEBUG=-O3 
# DEBUG=-O0 -g
CFLAGS=-Wall -I. $(NN_INCLUDES) $(OPTIONS) $(DEBUG)
NN_DEPS = $(wildcard *.h) $(wildcard ../common/*h) $(wildcard ../NNm/*) $(wildcard ../CNN/*h)

DEPS = Makefile $(HDEPS) $(NN_DEPS)

show: show.cc $(DEPS)
	$(CC) -o $@ $(CFLAGS) show.cc

checkGradient: checkGradient.cc $(DEPS)
	$(CC) -o $@ $(CFLAGS) checkGradient.cc -D__SAVE_G_FOR_VERIFICATION

ensemble: ensemble.cc $(DEPS)
	$(CC) -o $@ $(CFLAGS) ensemble.cc

all: show checkGradient ensemble

clean:
	rm show
	rm checkGradient
	rm ensemble
