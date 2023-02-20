# Creating slow-motion echocardiography
- The purpose of repositroy: sharing the codes for creating slow-motion echocardiography

As you might know, ischemic heart disease remains a major cause of mortality worldwide, despite a growing body of evidence regarding the development of appropriate treatments and preventive measures. Echocardiography has a pivotal role to diagnose cardiovascular disease, evaluate severity, determine treatment options, and predict prognosis. 

Stress echocardiography is applied to evaluate the presence of myocardial ischemia or the severity of valvular disease using some drugs or during exercise. However, diagnostic accuracy depends on the physicians’ experience and image quality due to its high-rate image. We assume that the optimization of video frame rate with the same image quality might contribute to improved evaluation of echocardiography in a difficult setting including evaluation for patients with very fast heart rate. 

Reference:  

**Super-SloMo** [![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)
PyTorch implementation of "Super SloMo: High Quality Estimation of Multiple Intermediate Frames for Video Interpolation" by Jiang H., Sun D., Jampani V., Yang M., Learned-Miller E. and Kautz J. [[Project]](https://people.cs.umass.edu/~hzjiang/projects/superslomo/) [[Paper]](https://arxiv.org/abs/1712.00080)


Slow-motion Echocardiography:<br/>Arbitrary time interpolation using echocardiographic images
------------------------------------------------------------------------------
For details, see the accompanying paper (medRxiv),

> [**XXXXXX**](https://XXXXXXXXXXXX)<br/>
  David Ouyang, Bryan He, Amirata Ghorbani, Neal Yuan, Joseph Ebinger, Curt P. Langlotz, Paul A. Heidenreich, Robert A. Harrington, David H. Liang, Euan A. Ashley, and James Y. Zou. <b>Nature</b>, March 25, 2020. https://doi.org/10.1038/s41586-020-2145-8


Dataset
-------
In this paper, all echocardiographic data were acquired using GE ultrasound equipment (see paper).
We are very sorry that data are not shared for privacy purposes. Please put your DICOM format echocardiographic data in XXX.
More than 500 echocardiographic video data obtained from about 100 patients were trained into this paper.
The authors used a GPU (GeForce Titan, 24GB) for training and inference.

## Prerequisites
This codebase was developed and tested with pytorch XX and CUDA XX and Python 3.6.

Install:
* [PyTorch](https://pytorch.org/get-started/previous-versions/)

For GPU, run
```bash
conda install pytorch=0.4.1 cuda92 torchvision==0.2.0 -c pytorch
```

* [TensorboardX](https://github.com/lanpa/tensorboardX) for training visualization
* [tensorflow](https://www.tensorflow.org/install/) for tensorboard
* [matplotlib](https://matplotlib.org/users/installing.html) for training graph in notebook.
* [tqdm](https://pypi.org/project/tqdm/) for progress bar in [video_to_slomo.py](video_to_slomo.py)
* [numpy](https://scipy.org/install.html)

EchoNet-Dynamic is implemented for Python 3, and depends on the following packages:
  - NumPy
  - PyTorch
  - Torchvision
  - OpenCV
  - skimage
  - sklearn
  - tqdm


Examples
--------
We show examples of our slow-motion echocardiography for nine(XXX) distinct patients below.
One subject has normal cardiac function, another has a regional wall motion abnormalities.
No human tracings for these patients were used by EchoNet-Dynamic.

| Time speed                                 | Normal                  | Regional wall motion abnormalities                            |
| ------                                 | ---------------------                  | ----------                             |
| Original | ![](docs/media/0X129133A90A61A59D.gif) | ![](docs/media/0X132C1E8DBB715D1D.gif) |
| 0.5x | ![](docs/media/0X13CE2039E2D706A.gif ) | ![](docs/media/0X18BA5512BE5D6FFA.gif) |
| 0.25x | ![](docs/media/0X16FC9AA0AD5D8136.gif) | ![](docs/media/0X1E12EEE43FD913E5.gif) |
| 0.125x | ![](docs/media/0X16FC9AA0AD5D8136.gif) | ![](docs/media/0X1E12EEE43FD913E5.gif) |

Installation
------------

First, clone this repository and enter the directory by running:

    git clone https://github.com/echonet/dynamic.git
    cd dynamic


Echonet-Dynamic and its dependencies can be installed by navigating to the cloned directory and running

    pip install --user .

Usage
-----
### Preprocessing DICOM Videos

The input of EchoNet-Dynamic is an apical-4-chamber view echocardiogram video of any length. The easiest way to run our code is to use videos from our dataset, but we also provide a Jupyter Notebook, `ConvertDICOMToAVI.ipynb`, to convert DICOM files to AVI files used for input to EchoNet-Dynamic. The Notebook deidentifies the video by cropping out information outside of the ultrasound sector, resizes the input video, and saves the video in AVI format. 

### Setting Path to Data

By default, EchoNet-Dynamic assumes that a copy of the data is saved in a folder named `a4c-video-dir/` in this directory.
This path can be changed by creating a configuration file named `echonet.cfg` (an example configuration file is `example.cfg`).

### Running Code

EchoNet-Dynamic has three main components: segmenting the left ventricle, predicting ejection fraction from subsampled clips, and assessing cardiomyopathy with beat-by-beat predictions.
Each of these components can be run with reasonable choices of hyperparameters with the scripts below.
We describe our full hyperparameter sweep in the next section.

#### Frame-by-frame Semantic Segmentation of the Left Ventricle

    echonet segmentation --save_video

This creates a directory named `output/segmentation/deeplabv3_resnet50_random/`, which will contain
  - log.csv: training and validation losses
  - best.pt: checkpoint of weights for the model with the lowest validation loss
  - size.csv: estimated size of left ventricle for each frame and indicator for beginning of beat
  - videos: directory containing videos with segmentation overlay

#### Prediction of Ejection Fraction from Subsampled Clips

  echonet video

This creates a directory named `output/video/r2plus1d_18_32_2_pretrained/`, which will contain
  - log.csv: training and validation losses
  - best.pt: checkpoint of weights for the model with the lowest validation loss
  - test_predictions.csv: ejection fraction prediction for subsampled clips

#### Beat-by-beat Prediction of Ejection Fraction from Full Video and Assesment of Cardiomyopathy

The final beat-by-beat prediction and analysis is performed with `scripts/beat_analysis.R`.
This script combines the results from segmentation output in `size.csv` and the clip-level ejection fraction prediction in `test_predictions.csv`. The beginning of each systolic phase is detected by using the peak detection algorithm from scipy (`scipy.signal.find_peaks`) and a video clip centered around the beat is used for beat-by-beat prediction.

### Hyperparameter Sweeps

The full set of hyperparameter sweeps from the paper can be run via `run_experiments.sh`.
In particular, we choose between pretrained and random initialization for the weights, the model (selected from `r2plus1d_18`, `r3d_18`, and `mc3_18`), the length of the video (1, 4, 8, 16, 32, 64, and 96 frames), and the sampling period (1, 2, 4, 6, and 8 frames).
