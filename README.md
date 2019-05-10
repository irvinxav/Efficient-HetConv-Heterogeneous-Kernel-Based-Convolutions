# Efficient HetConv: Heterogeneous Kernel-Based Convolutions 
Efficient HetConv: Heterogeneous Kernel-Based Convolutions for Deep CNNs
# Introduction
This is an efficient implementation of HetConv: Heterogeneous Kernel-Based Convolutions for Deep CNNs (CVPR 2019: https://arxiv.org/abs/1903.04120). HetConv reduces the computation (FLOPs) and the number of parameters as compared to standard convolution operation without sacrificing accuracy.
# Implementation
HetConv can be implemented in the following ways:
![alt text](https://github.com/irvinxav/Efficient-HetConv-Heterogeneous-Kernel-Based-Convolutions/blob/master/img/2.png)
This type of implementation can be found here
\
https://github.com/sxpro/HetConvolution2d_pytorch/.
\
But since it uses a loop structure, therefore it will slow down execution. Execution time will be more in this implementation.
![alt text](https://github.com/irvinxav/Efficient-HetConv-Heterogeneous-Kernel-Based-Convolutions/blob/master/img/1.png)
I implemented HetConv using the approach just shown above. The two implementations are the same (by reordering you can convert one to another). This type of implementation is more efficient. Discussion on the equivalence of the above two figures can be found 
here
\
1. https://github.com/sxpro/HetConvolution2d_pytorch/issues/3#issue-436278727
\
2. https://github.com/zhang943/Approximate-HetConv/issues/1#issuecomment-486956551
\
HetConv uses M/P kernels of size 3x3 and remaining (M−M/P) kernels are of size 1×1 as shown in the figure.
P is the part and M is the input depth (number of input channels).
I use group convolution to compute the result of M/P kernels of size 3x3 in each filter with group=p. 
For remaining (M−M/P) kernels are of size 1×1, I use a trick here:
I first compute the 1x1 convolution on all M input channels rather than the specific (M−M/P) kernels. Then in the second step, I make corresponding extra M/P 1x1 kernels weights to zero and masking the corresponding gradients so that extra M/P 1x1 kernels weights remains zero during backpropagations. This way we can implement original HetConv.
