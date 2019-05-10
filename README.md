# Efficient-HetConv-Heterogeneous-Kernel-Based-Convolutions
Efficient HetConv: Heterogeneous Kernel-Based Convolutions for Deep CNNs
# Introduction
This is an efficient implementation of HetConv: Heterogeneous Kernel-Based Convolutions for Deep CNNs (CVPR 2019: https://arxiv.org/abs/1903.04120). HetConv reduces the computation (FLOPs) and the number of parameters as compared to standard convolution operation without sacrificing accuracy.
# Implementation
HetConv can be implemented by the follwing ways:
![alt text](https://github.com/irvinxav/Efficient-HetConv-Heterogeneous-Kernel-Based-Convolutions/blob/master/img/2.png)
This type of implementation can be found here
\
https://github.com/sxpro/HetConvolution2d_pytorch/.
\
But since it uses loop structure therefore it will slow down execution. Execution time will be more in this implementation.
![alt text](https://github.com/irvinxav/Efficient-HetConv-Heterogeneous-Kernel-Based-Convolutions/blob/master/img/1.png)
We implemented HetConv using the approach just shown above. Acutally the two implenetations are exactly same (by reordering you can convert one to another). This type of implementation is more efficient. Discussion on the equivalence of above two figures can be found 
here
\
1. https://github.com/sxpro/HetConvolution2d_pytorch/issues/3#issue-436278727
\
2. https://github.com/zhang943/Approximate-HetConv/issues/1#issuecomment-486956551
\
HetConv uses M/P kernels of size 3x3 and remaining (M−M/P) kernels are of size 1×1 as shown in figure.
P is the part and M is the input depth (number of input channels).

