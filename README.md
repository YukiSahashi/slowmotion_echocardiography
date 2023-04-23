# Creating slow-motion echocardiography
- The purpose of repositroy: sharing the codes for creating slow-motion echocardiography

As you know, ischemic heart disease remains a major cause of mortality worldwide, despite a growing body of evidence regarding the development of appropriate treatments and preventive measures. Echocardiography has a pivotal role to diagnose cardiovascular disease, evaluate severity, determine treatment options, and predict prognosis. 

Stress echocardiography is applied to evaluate the presence of myocardial ischemia or the severity of valvular disease using some drugs or during exercise. However, diagnostic accuracy depends on the physicians’ experience and image quality due to its high-rate image. We assume that the optimization of video frame rate with the same image quality might contribute to improved evaluation of echocardiography in a difficult setting including evaluation for patients with very fast heart rate. 

Reference:  **Super-SloMo** (https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)
PyTorch implementation of "Super SloMo: High Quality Estimation of Multiple Intermediate Frames for Video Interpolation" by Jiang H., Sun D., Jampani V., Yang M., Learned-Miller E. and Kautz J. [[Project]](https://people.cs.umass.edu/~hzjiang/projects/superslomo/) [[Paper]](https://arxiv.org/abs/1712.00080)

![SuperSlomo_NVIDIA_paper](https://user-images.githubusercontent.com/58348086/231926369-7d347036-fcd0-49e0-ab11-3eb6ce0e456a.png)


Acknowldgement:
The project members sincerely appreciate the previous work (Super-Slomo) by NVIDIA. 


Slow-motion Echocardiography:<br/>Arbitrary time interpolation using echocardiographic images
------------------------------------------------------------------------------
For details, see the accompanying paper.<br/>

 (Submitted as of May 2023）<br/>
 Development of artificial intelligence-based slow-motion echocardiography and clinical usefulness for evaluating regional wall motion abnormalities
> [**XXXXXX**](https://XXXXXXXXXXXX)　
PDFをここにおく<br/>


Authors: <br/>
Yuki Sahashi MD*, MSc, Takeshita Ryo*, Takatomo Watanabe MD, PhD, Takuma Ishihara PhD, Ayako Sekine, Daichi Watanabe, Takeshi Ishihara MD, Hajime Ichiryu MD, Susumu Endo, Daisuke Fukuoka PhD, Takeshi Hara Ph, Hiroyuki Okura MD PhD, 

Dataset
-------
In this paper, all echocardiographic data were acquired using GE ultrasound equipment. 
We are very sorry that data are not shared for privacy purposes. 
<br/>
Please put your DICOM format echocardiographic data in XXX.
More than 1300 echocardiographic video data obtained from about 120 patients were trained into this paper.
The authors used a GPU (GeForce Titan, 24GB) for training and inference.
Please use your own dataset to create the arbitrary-time slow-motion echocardiography.



## Prerequisites
Model implementation and statistical analysis were performed using the Python (version 3.8) with Pytorch (version. 1.8.0) and R (version. 3.4.1), respectively. Model training was conducted using a graphics processing unit (GeForce Titan RTX 24 GB, NVIDIA, Santa Clara, California).

This project is implemented for Python 3, and depends on the following packages:
  - NumPy
  - PyTorch
  - Torchvision
  - sklearn
  - tqdm
  - tensorflow for tensorboard
  - etc.


Examples
--------
We show examples of our slow-motion echocardiography below. One subject has normal cardiac function, another has a slight regional wall motion abnormalities.

| Time speed                                 | Normal                  | Regional wall motion abnormalities                            |
| ------                                 | ---------------------                  | ----------                             |
| Original | ![](docs/media/0X129133A90A61A59D.gif) | ![](docs/media/0X132C1E8DBB715D1D.gif) |
| 0.5x | ![](docs/media/0X13CE2039E2D706A.gif ) | ![](docs/media/0X18BA5512BE5D6FFA.gif) |
| 0.25x | ![](docs/media/0X16FC9AA0AD5D8136.gif) | ![](docs/media/0X1E12EEE43FD913E5.gif) |
| 0.125x | ![](docs/media/0X16FC9AA0AD5D8136.gif) | ![](docs/media/0X1E12EEE43FD913E5.gif) |


Usage
-----
## Video Converter
First, clone this repository, then please do this.
You can convert any echocardiography video to a slomo or high fps video using [video_to_slomo.py](video_to_slomo_SF.py). Use the command

```bash

# Windows
python video_to_slomo.py --ffmpeg path\to\folder\containing\ffmpeg --video path\to\video.mp4 --sf N --checkpoint path\to\checkpoint.ckpt --fps M --output path\to\output.mkv

# Linux
python video_to_slomo.py --video path\to\video.mp4 --sf N --checkpoint path\to\checkpoint.ckpt --fps M --output path\to\output.mkv
```
