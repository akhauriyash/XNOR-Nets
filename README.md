# XNOR kernels on Intel ISA
Attempting to implement XNOR neural network functions (Convolution, GEMM ...) on Intel ISA.

For CUDA compatiable (Nvidia GPU) XNOR convolutional kernel, check out [this repository.](https://github.com/akhauriyash/XNOR-convolution)

## xGEMM (Binarized General Matrix Multiply on Intel Xeon Phi)

Run xCMMAbench.c for benchmarking the algorithm.

**Important note: Here, the binarization time of the second matrix (B) is not taken into consideration.
                  This is due to the fact that these will be pre binarized in the network. 
                  We have also ignored the _considerable_ savings (as we scale up) we will have by pre-storing
                  the _transposed_ binarized B matrix, as this can be said for full precision CMMA algorithm as well,
                  but to a lesser extent. (Essentially, the xCMMA algorithm should scale better as well.)**
                  
## xCONV (Binarized convolution on Intel Xeon Phi)

Run xCONVbench.c for benchmarking the algorithm.

**Note that both images have a logarithmic left vertical-axis.**

## Benchmarks
In progress

## To run:
   These codes have been written on the Colfax Cluster optimized for **Intel Xeon Phi KNL 7210.**
   To run xCMMA.c, execute this:
   on Xeon Phi KNL 7210
   `icpc -xMIC-AVX512 -qopenmp -mkl -fp-model fast=2 -fma -unroll=4 xCMMA.c -o xCMMA.out && echo ~/parallel/xCMMA.out | qsub`
   on Xeon Gold 6128
   `icpc -xCORE-AVX512 -qopenmp -mkl -fp-model fast=2 -fma -unroll=4 xCMMA.c -o xCMMA.out && echo ~/parallel/xCMMA.out | qsub`
  
##  Hardware specifications:
  * Intel® Xeon Phi™ 7210 Processor
  * Intel® Xeon Gold 6128 Processor
    
##  Note:
  This is a work in progress. There might be some mistakes here. 
  Do let me know if you find any logical errors in the code.
 
##  TO DO:
  - [ ] Upload codes
  - [ ] Update code gists
  - [ ] Create CLI
  - [ ] Create function timers
  - [ ] Link main function with parser
