# PROCA: Place Recognition under Occlusion and Changing Appearance via Disentangled Representations 

**[Paper](https://arxiv.org/pdf/2211.11439.pdf) |
[Video](https://www.youtube.com/watch?v=W_tol4aHIQk)**

[Yue Chen¹](https://scholar.google.com/citations?user=M2hq1_UAAAAJ&hl=en), 
[Xingyu Chen¹†⚑](https://rover-xingyu.github.io/),
[Yicen Li²](https://yicen-research.webador.com/)

[†Corresponding Author](https://rover-xingyu.github.io/), 
[⚑Project Lead](https://rover-xingyu.github.io/),
[¹Xi'an Jiaotong University](http://en.xjtu.edu.cn/),
[²McMaster University](https://www.mcmaster.ca/)

This repository is an official implementation of PROCA using [pytorch](https://pytorch.org/). 

[![PROCA: Place Recognition under Occlusion and Changing Appearance via Disentangled Representations](https://res.cloudinary.com/marcomontalbano/image/upload/v1678023918/video_to_markdown/images/youtube--W_tol4aHIQk-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=W_tol4aHIQk "PROCA: Place Recognition under Occlusion and Changing Appearance via Disentangled Representations")

## Usage

### Prerequisites
- Python 3
- Pytorch and torchvision (https://pytorch.org/)
- [TensorboardX](https://github.com/lanpa/tensorboard-pytorch)

### Install
- Clone this repo:
```
git clone https://github.com/rover-xingyu/PROCA.git
cd PROCA
```

### Datasets
- The dataset we use in this paper is from [CMU-Seasons Dataset](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/CMU-Seasons/).
- In order to verify the generalization ability of PROCA, we sample images from the urban part as the training set, and evaluate on the suburban and park parts. 
- We label the images with occlusion and without occlusion depending on if there are dynamic objects in the images. You can find our labels in [`dataset/CMU_Seasons_Occlusions.json`](dataset/CMU_Seasons_Occlusions.json) 
- The dataset is organized as follows:
```
    ├── CMU_urban
    │   ├── trainA // images with appearance A without occlusion
    │   │   ├── img_00119_c1_1303398474779487us_rect.jpg
    │   │   ├── ...
    │   ├── trainAO // images with appearance A with occlusion
    │   │   ├── img_00130_c0_1303398475779409us_rect.jpg
    │   │   ├── ...
    │   ├── ...
    │   │   trainL // images with appearance L without occlusion
    │   │   ├── img_00660_c0_1311874734447600us_rect.jpg
    │   │   ├── ...
    │   ├── trainLO // images with appearance L with occlusion
    │   │   ├── img_00617_c1_1311874730447615us_rect.jpg
    │   │   ├── ...
```

## Citation

If you find this project useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{chen2023place,
  title={Place Recognition under Occlusion and Changing Appearance via Disentangled Representations},
  author={Chen, Yue and Chen, Xingyu and Li, Yicen},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={1882--1888},
  year={2023},
  organization={IEEE}
}
```

## Acknowledge
Our code is based on the awesome pytorch implementation of Diverse Image-to-Image Translation via Disentangled Representations ([DRIT++](https://github.com/HsinYingLee/DRIT) and [MDMM](https://github.com/HsinYingLee/MDMM)). We appreciate all the contributors.
