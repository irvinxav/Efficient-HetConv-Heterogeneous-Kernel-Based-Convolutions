# Efficient HetConv: Heterogeneous Kernel-Based Convolutions 
Efficient HetConv: Heterogeneous Kernel-Based Convolutions for Deep CNNs
# Introduction
This is an efficient implementation of HetConv: Heterogeneous Kernel-Based Convolutions for Deep CNNs (CVPR 2019: https://arxiv.org/abs/1903.04120). HetConv reduces the computation (FLOPs) and the number of parameters as compared to standard convolution operation without sacrificing accuracy.
# Implementation
HetConv can be implemented in the following ways:
![alt text](https://github.com/irvinxav/Efficient-HetConv-Heterogeneous-Kernel-Based-Convolutions/blob/master/img/2.png)<br/>
This type of implementation can be found here<br/>

https://github.com/sxpro/HetConvolution2d_pytorch/.<br/>

But since it uses a loop structure, therefore it will slow down execution. Execution time will be more in this implementation.
![alt text](https://github.com/irvinxav/Efficient-HetConv-Heterogeneous-Kernel-Based-Convolutions/blob/master/img/1.png)<br/>
I implemented HetConv using the approach just shown above. The two implementations are the same (by reordering you can convert one to another). This type of implementation is more efficient. Discussion on the equivalence of the above two figures can be found 
here<br/>

1. https://github.com/sxpro/HetConvolution2d_pytorch/issues/3#issue-436278727<br/>

2. https://github.com/zhang943/Approximate-HetConv/issues/1#issuecomment-486956551<br/>

HetConv uses M/P kernels of size 3x3 and remaining (M−M/P) kernels are of size 1×1 as shown in the figure.
P is the part and M is the input depth (number of input channels).
I use group convolution to compute the result of M/P kernels of size 3x3 in each filter with group=p. 
For remaining (M−M/P) kernels are of size 1×1, I use a trick here:
I first compute the 1x1 convolution on all M input channels rather than the specific (M−M/P) kernels. Then in the second step, I make corresponding extra M/P 1x1 kernels weights to zero and masking the corresponding gradients so that extra M/P 1x1 kernels weights remains zero during backpropagations. This way we can implement original HetConv.

# Experiments
By changing "part" P value in the code (hetconv.py), I reproduced the results for VGG-16 on CIFAR-10 in different setups.

| __Model__ | __FLOPs__ | __Acc% (Original)__ | __Acc% (Reproduced)__ |
|-------------|------------|------------|------------|
| VGG-16_P1   | 313.74M     | 94.06     | 94.1      |
| VGG-16_P2   | 175.23M     | 93.89     | 93.9      |
| VGG-16_P4   | 105.98M     | 93.93     | 93.9     |
| VGG-16_P8   | 71.35M     | 93.92     | 93.9      |
| VGG-16_P16   | 54.04M     | 93.96    | 93.9     |
| VGG-16_P32   | 45.38M     | 93.73     | 93.8      |
| VGG-16_P64   | 41.05M     | 93.42     | 93.4      |


# Future work
I have implemented HetConv using group wise and point wise convolution, but it can also be implemented directly.
The direct implementation of HetConv written in CUDA will further increase the speed and efficiency.<br/>
Alternatively, we can make corresponding extra M/P 1x1 kernels weights to zero in point wise convolution and then using sparse convolution for point wise convolution, also result in an improvement in practical speed further. In this types of implementation gradients masking will not be required since we are using sparse point wise convolution. 
