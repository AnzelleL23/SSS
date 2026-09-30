# SSS
Code for Semi-stochastic Sinkhorn experiments.

## Brief Description of Files
- (1) SSS.ipynb: All the code that we used to produce all the figures, graphs and tables in our paper can be ran directly from here. We use several OT functions from the POT library https://pythonot.github.io/. Our implementations for the Sinkhorn baseline methods and semi-dual methods are adapted from their respective codes in the POT library as well. For our experiments involving the quantitative Sinkhorn bridge, we adapt the code directly from https://github.com/APooladian/SinkhornBridges.
- (2) sss_solvers.py: contains the CPU and GPU implementations of SSS for direct use in external code.

## How to use
### 1. SSS.ipynb
#### Importing modules
Most of the modules and libraries are already pre-installed in Google Colab, the only library that requires additional installation is POT.

#### Sections
Here is an overview of the different sections of our notebook. 
- (1) Verifying GS--Jacobi relationship
- (2) Comparison with Sinkhorn baselines: These experiments are done on CPU only.
    - (a) 2D EOT data
    - (b) Color transfer
    - (c) Sinkhorn Bridge: We use the code from https://github.com/APooladian/SinkhornBridges here. 
- (3) SSS minibatch
- (4) Appendix extra experiments
    - (a) Varying $n/m$
    - (b) Minibatch ablation
    - (c) Comparison with Semi-dual schemes
 
Most sections can be ran independently, although for some cells requiring functions from previous ones, we note which cells to run in comments at the top of the cell. CPU experiments can take a while to run, especially when using higher computational budgets ($10^8$ and above). Most GPU experiments run quickly, except for those concerning the minibatch ablation. 

### 2. sss_solvers.py
This file provides self contained CPU and GPU implementations of SSS. The code can be used directly, or integrated into external code. Note that our implementation is based on Algorithm 3 on page 36 of our paper. 
