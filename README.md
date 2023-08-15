# UniTR: The First Unified Multi-modal Transformer Backbone for 3D Perception

This repo is the official implementation of ICCV paper: [UniTR: A Unified and Efficient Multi-Modal Transformer for Bird's-Eye-View Representation](https://github.com/Haiyang-W/UniTR) as well as the follow-ups. Our UniTR achieves state-of-the-art performance on nuScenes Dataset with a real unified and weight-sharing multi-modal (e.g., `Cameras` and `LiDARs`) backbone. We have made every effort to ensure that the codebase is clean, concise, easily readable, state-of-the-art, and relies only on minimal dependencies.

<div align="center">
  <img src="assets/Figure1.png" width="700"/>
</div>

> UniTR: A Unified and Efficient Multi-Modal Transformer for Bird's-Eye-View Representation
>
> [Haiyang Wang*](https://scholar.google.com/citations?user=R3Av3IkAAAAJ&hl=en&oi=ao), [Hao Tang*](https://scholar.google.com/citations?user=MyarrsEAAAAJ&hl=en), Shaoshuai Shi $^\dagger$, Aoxue Li, Zhenguo Li, Bernt Schiele, Liwei Wang $^\dagger$
> 
> Contact: Haiyang Wang (wanghaiyang6@stu.pku.edu.cn), Hao Tang (tanghao@stu.pku.edu.cn), Shaoshuai Shi (shaoshuaics@gmail.com)

🔥 Gratitude to Tang Hao for extensive code refactoring and noteworthy contributions to open-source initiatives. His invaluable efforts were pivotal in ensuring the seamless completion of UniTR.



## News
- [23-07-13] 🔥 UniTR is accepted at [ICCV 2023](https://iccv2023.thecvf.com/).
- [23-08-15] UniTR is released on [arXiv](https://github.com/Haiyang-W/UniTR).

## Overview
- [Todo](https://github.com/Haiyang-W/UniTR#todo)
- [Introduction](https://github.com/Haiyang-W/UniTR#introduction)
- [Main Results](https://github.com/Haiyang-W/UniTR#main-results)
- [Quick Start](https://github.com/Haiyang-W/UniTR#quick-start)
- [Citation](https://github.com/Haiyang-W/UniTR#citation)
- [Acknowledgments](https://github.com/Haiyang-W/UniTR#potential-research)

## TODO

- [x] Release the [arXiv](https://github.com/Haiyang-W/UniTR) version.
- [x] SOTA performance of multi-modal 3D object detection (Nuscenes) and BEV Map Segmentation (Nuscenes).
- [ ] Clean up and release the code of NuScenes (before ICCV main conference).
- [ ] Merge UniTR to [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

## Introduction
Jointly processing information from multiple sensors is crucial to achieving accurate and robust perception for reliable autonomous driving systems. However, current 3D perception research follows a modality-specific paradigm, leading to additional computation overheads and inefficient collaboration between different sensor data. 
<div align="center">
  <img src="assets/Figure2.png" width="500"/>
</div>

In this paper, we present an efficient multi-modal backbone for outdoor 3D perception, which processes a variety of modalities with unified modeling and shared parameters. It is a fundamentally task-agnostic backbone that naturally supports different 3D perception tasks. It sets a new state-of-the-art performance on the nuScenes benchmark, achieving `+1.1 NDS` higher for 3D object detection and `+12.0 mIoU` higher for BEV map segmentation with lower inference latency.
<div align="center">
  <img src="assets/Figure3.png" width="800"/>
</div>

## Main results
### 3D Object Detection (on NuScenes validation)
|  Model  | NDS | mAP |mATE | mASE | mAOE | mAVE| mAAE | ckpt | Log |
|---------|---------|--------|---------|---------|--------|---------|--------|--------|--------|
|  UniTR | 73.1 | 70.0 | 26.3 | 24.7 | 26.8 | 24.6 | 17.9 | [ckpt](https://github.com/Haiyang-W/UniTR)| [Log](https://github.com/Haiyang-W/UniTR)|
|  UniTR+LSS | 73.3 | 70.5 | 26.0 | 24.4 | 26.8 | 24.8 | 18.7 | [ckpt](https://github.com/Haiyang-W/UniTR)| [Log](https://github.com/Haiyang-W/UniTR)|


### 3D Object Detection (on NuScenes test)
|  Model  | NDS | mAP | mATE | mASE | mAOE | mAVE| mAAE |
|---------|---------|--------|--------|---------|---------|--------|---------|
|  UniTR | 74.1 | 70.5 | 24.4 | 23.3 | 25.7 | 24.1 | 13.0 |
|  UniTR+LSS | 74.5 | 70.9 | 24.1 | 22.9 | 25.6 | 24.0 | 13.1 |

### Bev Map Segmentation (on NuScenes validation)
|  Model  | mIoU | Drivable |Ped.Cross.| Walkway |  StopLine  | Carpark |  Divider  |  ckpt | Log |
|---------|----------|--------|--------|--------|--------|---------|--------|---------|--------|
|  UniTR | 73.2  | 90.4   |   73.1   |   78.2   |   66.6   |   67.3  |   63.8   |  [ckpt](https://github.com/Haiyang-W/UniTR)| [Log](https://github.com/Haiyang-W/UniTR)|
|  UniTR+LSS |74.7 |   90.7   |   74.0   |   79.3   |   68.2   |   72.9  |   64.2   | [ckpt](https://github.com/Haiyang-W/UniTR)| [Log](https://github.com/Haiyang-W/UniTR)|

### What's new here?
#### 🔥 Beats previous SOTAs of outdoor multi-modal 3D Object Detection and BEV Segmentation
Our approach has achieved the best performance on multiple tasks (e.g., 3D Object Detection and BEV Map Segmentation), and it is highly versatile, requiring only the replacement of the backbone.
##### 3D Object Detection
<div align="left">
  <img src="assets/Figure4.png" width="700"/>
</div>

##### BEV Map Segmentation
<div align="left">
  <img src="assets/Figure5.png" width="700"/>
</div>

#### 🔥 Weight-Sharing among all modalities 
We introduce a modality-agnostic transformer encoder to handle these view-discrepant sensor data for parallel modal-wise representation learning and automatic cross-modal interaction without additional fusion steps.

## Quick Start
- TODO: Kindly request your patience.
  
## Citation
Please consider citing our work as follows if it is helpful.
```
@inproceedings{wang2023unitr,
    title={UniTR: A Unified and Efficient Multi-Modal Transformer for Bird's-Eye-View Representation},
    author={Haiyang Wang, Hao Tang, Shaoshuai Shi, Aoxue Li, Zhenguo Li, Bernt Schiele, Liwei Wang},
    booktitle={ICCV},
    year={2023}
}
```

## Acknowledgments
UniTR uses code from a few open source repositories. Without the efforts of these folks (and their willingness to release their implementations), UniTR would not be possible. We thanks these authors for their efforts!
* Shaoshuai Shi: [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
* Chen Shi: [DSVT](https://github.com/Haiyang-W/DSVT)
* Zhijian Liu: [BevFusion](https://github.com/mit-han-lab/bevfusion)