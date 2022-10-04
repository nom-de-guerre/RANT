
This is the Another Neural Toy (ANT) project.  It is an implementation of
neural network training and support libraries in C++.  It includes support
for regression and classification.  Classification can be either multiclass
or multilabel.

There are 3 directories.  

NNm: The ANT library.  The code that implements the core system is found here.
   NNm.h and NNm.tcc are the core system implementing NNet_t.  The remaining
   files implement update strategies (ADAM and RPROP+) and layers.  Only
   dense feed-forward ANNs are supported.  The fully connected (dense) 
   restriction can be overcome by entering zeros in a weight matrix.

common: miscellaeous headers implementing ancillary functionality.

Examples: A directory with a number of examples.  The examples exhibit
   regression, multiclass classification and multilabel classification.

The data files that the examples depend are documented in the individual
files.

There are 4 activation functions available.  The default is the sigmoid.  To
override this behaviour define one of the following at compile time:

	(default): sigmoid: x --> [0, 1]

	__TANH_ACT_FN : tanh activation function R: --> [-1, 1]

	__RELU : max (0, x)

	__IDENTITY : x --> x

The Makefile in Examples accepts compile time arguments with OUTSIDE.

Activation functions are specified at compile time for run-time 
performance.  iris.cc runs in 0.04 seconds ({20, 20} dense ANN), Keras
takes 25 seconds: 625x faster (M1 SoC).

Keras is more flexible, but for certain applications the C++ CPU optimized
approach makes sense.

