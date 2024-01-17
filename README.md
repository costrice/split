# SPLiT

This is the repository for the official implementation and project page of the
paper [SPLiT: Single Portrait Lighting Estimation via a Tetrad of Face Intrinsics](https://www.computer.org/csdl/journal/tp/2024/02/10301699/1RFBJBi1aHS).

<p align="center"> 
<img src="docs/static/images/teaser.png">
</p>

SPLiT estimates a tetrad of face intrinsics and uses spherically distributed
components to estimate lighting.

---

## [Project page](https://costrice.github.io/split/) | [Paper](https://camera.pku.edu.cn/Fei_TPAMI23.pdf) | [Data](#SPLiT-Faces-and-Light-Dataset)

## Getting Started

### Required Dependencies

To run the code successfully, the following dependencies are required:

* Python 3
* PyTorch
* Torchvision
* OpenCV (cv2)
* antialiased-cnns
* tqdm
* yacs
* mediapipe (only for face detection)

For example, you can install the dependencies using pip:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python antialiased-cnns tqdm yacs mediapipe
```

You will also need a CUDA-enabled GPU to run the code.

### Tested Configurations

The inference code has been successfully tested on the following configurations:

* Windows 11, Python 3.9.18, PyTorch 2.1.2 with CUDA 11.8, Torchvision 0.16.2
* GPU: NVIDIA GeForce RTX 3090 (24GB)

### Inference

#### Data Preparation

Put the portrait images to be processed in the `/images/face` folder.
If there are any face region masks available, put them in the `/images/mask`
folder with the same name as the corresponding portrait image and with the
suffix `.png`.

We have put 2 portrait images from Internet as examples.

#### Pretrained Models

We provide pretrained models of SPLiT, which can be downloaded from:

* [PKU disk link](https://disk.pku.edu.cn/#/link/E2AE5C46989B95D72D44C138610CE928)

Users should put all model files into the `/model_files/` folder before running
inference.

#### Step 1: Pre-processing

This step finds all image files in the `/images/face/` folder, and for each
image, crops the face region from the portrait image and generates the face
region mask (without mouth and eyes), if not provided.
If the face region mask is provided, the face image will not be
cropped, and the face region mask will be used directly.
Otherwise, we use [mediapipe](https://github.com/google/mediapipe) to detect
faces and the face semantic segmentation model
in [FFHQ-Aging-Dataset](https://github.com/royorel/FFHQ-Aging-Dataset) to
generate the face region mask.

```bash
python 1_preprocess.py
```

The results are saved into the `/images/pre_processed/` folder.
Each portrait image produces 3 files:

* `*.png`: the cropped face image.
* `*_mask.png`: the face region mask.
* `*_masked.png`: the masked face image.

Note that the automatic face region mask generation may not be
very accurate. But it is sufficient for lighting estimation.

#### Step 2: Run Intrinsic Decomposition

Tetrad Intrinsic Decomposition module takes the cropped face images and
face region masks as input, and predicts the four face intrinsics.
After that, the face intrinsics are distributed onto spheres using the surface
normal as indices.

```bash
python 2_intrinsics_decomp.py
```

The results are saved into the `/images/intrinsic/` folder.
Each portrait image produces 8 files:

* `*_intrA.png`: the diffuse albedo in gamma-corrected sRGB color space.
* `*_intrN.png`: the surface normal.
* `*_intrD.hdr`: the HDR diffuse shading.
* `*_intrS.hdr`: the HDR specular reflection.
* `*_intrRecon.png`: the reconstructed face image (A*D+S).
* `*_distM.png`: the distributed mask.
* `*_distD.hdr`: the distributed diffuse shading.
* `*_distS.hdr`: the distributed specular reflection.

#### Step 3: Run Lighting Estimation

Spherical Lighting Estimation module takes distributed spherical face
components as input, and predicts high-dynamic-range environment lighting
in the form of mirror balls.

```bash
python 3_lighting_estim.py
```

Both indoor and outdoor lighting estimators are run in this step, and
the results are saved into the `/images/lighting/indoor/`
and `/images/lighting/outdoor/` folders, respectively.
Each portrait image produces 3 files for each scene category:

* `*_pred.hdr`: the estimated HDR environment lighting, which can be used to
  render other objects under the same lighting as the input portrait.
* `*_ibr_diff.png`: the image of an example diffuse ball rendered under the
  estimated lighting.
* `*_ibr_spec.png`: the image of an example glossy ball rendered under the
  estimated lighting.

### Training (Not Yet Released)

We have not yet released our training dataset, scripts to generate the training
set, and codes to train our model.

As our training dataset is built on
the [FaceScape](https://facescape.nju.edu.cn/) 3D face dataset, we can not
distribute our data which contains the re-rendered images of their subjects,
due to the license of FaceScape.
Thus, whoever wants to train our model from scratch needs to apply for the
FaceScape dataset and re-render the training images.
If you want to train our model, please contact us, and we will provide help as
much as we can.

## SPLiT Faces and Light Dataset

We captured a dataset containing pairs of real portrait images and the
corresponding HDR environment lighting.
This dataset contains 66 pairs of 9 human subjects in 11 outdoor scenes and 84
pairs of 10 human subjects in 9 indoor scenes.

To apply for this dataset, please first fill
the [End User Licence Agreement](/docs/static/pdfs/SPLiT_Faces_and_Light_Dataset_EULA.pdf)
and return it to Boxin Shi by email at <shiboxin@pku.edu.cn>.
Then we will send you the download link of the dataset.

## Change Log

* 2024/1/17: Release pre-trained models and testing real dataset
* 2024/1/16: First release (inference code)

## License

Code: under Apache License 2.0

Pretrained model file: according to license
of [FaceScape](https://facescape.nju.edu.cn/) which we used to
train our model, no commercial use is allowed.

## Citation

If you find SPLiT useful for your work, please consider citing:

```
@article{split2023,
    author = {Fan Fei and Yean Cheng and Yongjie Zhu and Qian Zheng and Si Li and Gang Pan and Boxin Shi},
    journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
    title = {{SPLiT}: Single Portrait Lighting Estimation via a Tetrad of Face Intrinsics},
    year = {2024},
    volume = {46},
    number = {02},
    issn = {1939-3539},
    pages = {1079-1092},
    doi = {10.1109/TPAMI.2023.3328453},
    publisher = {IEEE Computer Society},
    address = {Los Alamitos, CA, USA},
    month = {feb}
}
```

## Contacts

Please contact <feifan_eecs@pku.edu.cn> or open an issue for any questions or
suggestions.

Thanks! :smiley:
