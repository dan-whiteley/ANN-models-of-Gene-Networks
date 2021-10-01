# ANN-models-of-Gene-Networks
Gene Networks are responsible for creating 2D patterns of gene expression in development, here we use neural networks, of any structure, to model such pattern formation.

The implementation of generalised backpropagation is contatined in the updateWeights() function in the RNet.h header file.

The code can be run with a chosen number of nodes with >python parallel-handler.py, it's currently set up to just run one recurrent network with 15 nodes.

main.cpp is the core of the algorithm, it initialises the structure of the network, then trains the network to reproduce the 4 knockout images.
