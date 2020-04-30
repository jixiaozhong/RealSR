# RealSR 

Real-World Super-Resolution via Kernel Estimation and Noise Injection (CVPRW 2020)

Xiaozhong Ji, Yun Cao, Ying Tai, Chengjie Wang, Jilin Li, and Feiyue Huang

*Tencent YouTu Lab*

Our solution is the **winner of CVPR NTIRE 2020 Challenge on Real-World Super-Resolution** in both tracks.

(*Official PyTorch Implementation*)

## Introduction

Recent state-of-the-art super-resolution methods have achieved impressive performance on ideal datasets regardless of blur and noise. However, these methods always fail in real-world image super-resolution, since most of them adopt simple bicubic downsampling from high-quality images to construct Low-Resolution (LR) and High-Resolution (HR) pairs for training which may lose track of frequency-related details. To address this issue, we focus on designing a novel degradation framework for real-world images by estimating various blur kernels as well as real noise distributions. Based on our novel degradation framework, we can acquire LR images sharing a common domain with real-world images. Then, we propose a real-world super-resolution model aiming at better perception. Extensive experiments on synthetic noise data and real-world images demonstrate that our method outperforms the state-of-the-art methods, resulting in lower noise and better visual quality. In addition, our method is the winner of NTIRE 2020 Challenge on both tracks of Real-World Super-Resolution, which significantly outperforms other competitors by large margins. 

![RealSR](figures/arch.png)  

If you are interested in this work, please cite

    @article{Ji2020realsr,
        title={Real World Super-Resolution via Kernel Estimation and Noise Injection},
        author={Xiaozhong Ji, Yun Cao, Ying Tai, Chengjie Wang, Jilin Li, and Feiyue Huang},
        journal={CVPR Workshops},
        year={2020},
    }

 
## Visual Results

![0](figures/0913.png)

![1](figures/0935.png)

## Models and Training Codes
Release soon.