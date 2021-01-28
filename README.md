# STDNet-Crowd-Counting

This is the official Tensorflow implementation of the paper: "Spatiotemporal Dilated Convolution with Uncertain Matching for Video-based Crowd Estimation (T-MM 2021)".

[Paper Link](https://ieeexplore.ieee.org/document/9316927)


## Citation

If you use this code for your research, please cite our paper. Thank you!

```
@article{9316927,
  author={Yu-Jen Ma, Hong-Han Shuai,and Wen-Huang Cheng},
  journal={IEEE Transactions on Multimedia}, 
  title={Spatiotemporal Dilated Convolution with Uncertain Matching for Video-based Crowd Estimation}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMM.2021.3050059}
}
```

## Code

### Install Dependencies

The code is used with Python 3.6, requiring the packages listed below.

```
tensoflow==1.14.0
opencv-python
pillow
scipy
numpy
```
The packages can be easily installed by pip install.

### Train

1. Download the preprocessed UCSD Dataset and the initial weight for the VGG backbone. [Google Drive Link](https://drive.google.com/file/d/1_6ssL1b9nMvgMdK8Owea_7ksWxObgWeh/view?usp=sharing)

2. Unzip the downloaded file and modify the path to the same directory of this repository.

3. Run the python file (including the evaluation on the testing set).

  `python Train.py`


### The other details will be updated soon.

to be continued.
