There are two directories.  

NNm, which is a very simple implementation of a RPROP+ neural network.  It 
includes an example of regression and classification.  Make all will produce
sine and classify.  The commoand line for both accepts the network architecture.
The number of inputs are implicit as they are determined by the program.

CNN is a primitive convolutional network.  The example, digits.cc, trains
with the MNIST data set and then verifies.  It approximates the LeNet-5
CNN (C1, S2, C3, S4 and F5).  1:1 filter to maxpool, specified maxpool to 
filter, and fully connected to a classifier neural network.


