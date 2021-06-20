There are four directories.  

NNm, which is a very simple implementation of a RPROP+ neural network.  There
 are simple examples of its use in the Examples directory.  There are two
 examples: regression and classification.  Make all will produce
sine and classify.  The command line for both accepts the network architecture.
Try 4 3 1 for sine and 7 4 4 for classify.  The number of inputs are implicit
 as they are determined by the program.

CNN is a primitive convolutional network.  It includes a few examples that
train with the MNIST data set.  There are 3 examples.  digits.cc,
simpleProgram.cc and LeNet5.cc.  The latter is a crude approximation.  In
particular it uses Softmax, not radial functions, and C5 is a NN, not a
convolutional layer.

