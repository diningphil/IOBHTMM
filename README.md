# IOBHTMM

This repository provides the official Python implementation for the Bottom-up Hidden Tree Markov model (BHTMM) in its (more general) input-output version (IOBHTMM). The model learns a distribution over tree-structured data, implemented throughout a generative process acting from the leaves to the root of the tree.  

The library includes both a script to reproduce the tree classification experiments reported in the paper describing the method, as well as a more general configuration script showing how to use the model both in homogenous (BHTMM) and input-driven (IOBHTMM) version.

This research software is provided as is. If you happen to use or modify this code, please remember to cite the foundation papers:

Davide, Bacciu; Alessio, Micheli; Alessandro, Sperduti, Compositional Generative Mapping for Tree-Structured Data; Part I: Bottom-Up Probabilistic Modeling of Trees,  Neural Networks and Learning Systems, IEEE Transactions on, 23 (12), pp. 1987 -2002, 2012

Davide, Bacciu; Alessio, Micheli; Alessandro, Sperduti, An input–output hidden Markov model for tree transductions, Neurocomputing, 112 , pp. 34–46, 2013

If you have any query concerning the model (not its implementation), feel free to contact the corresponding Author of the paper (http://www.di.unipi.it/~bacciu/). Note that the code can be easily adapted to compute the generative Jaccard tree kernels described here:

Davide, Bacciu; Alessio, Micheli; Alessandro, Sperduti, Integrating bi-directional contexts in a generative kernel for trees. In: Neural Networks (IJCNN), 2014 International Joint Conference on. IEEE, 2014. p. 4145-4151.
