# SPLiT

This is the repository for the official implementation and project page of the
paper [SPLiT: Single Portrait Lighting Estimation via a Tetrad of Face Intrinsics](https://www.computer.org/csdl/journal/tp/2024/02/10301699/1RFBJBi1aHS).

<p align="center"> 
<img src="docs/static/images/teaser.png">
</p>

SPLiT estimates a tetrad of face intrinsics and uses spherically distributed
components to estimate lighting.

---

## [Project page](https://costrice.github.io/split/) | [Paper](https://camera.pku.edu.cn/Fei_TPAMI23.pdf) | Data

---

## Getting Started

### Install

We implement SPLiT using PyTorch.

```
pip install torch torchvision torchaudio mediapipe antialiased-cnns
```

(Not tested yet, should work)

### Inference

#### Data Preparation

Put the portrait images to be processed in the `/images/face` folder.
If there are any face region masks available, put them in the `/images/mask`
folder with the same name as the corresponding portrait image and with the
suffix `.png`.

We have put 2 portrait images from Internet as examples.

#### Pretrained Models

We provide pretrained models of SPLiT, which can be downloaded from [TBA].
Users should put the model files in `checkpoints/` before running inference.

#### Step 1: Pre-processing

This step finds all image files in the `/images/face` folder, and for each
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

The results will be put into the `/images/pre_processed/` folder.
Each portrait image will have three files: `*.png` which is the cropped
image, `*_mask.png` which is the face region mask,
, and `*_masked.png` which is the masked face image.

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

#### Step 3: Run Lighting Estimation

Spherical Lighting Estimation module takes distributed spherical face
components as input, and predicts high-dynamic-range environment lighting
in the form of mirror balls.

```bash
python 3_lighting_estim.py
```

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

## Change Log

* 2024/1/18 (expected): Release pre-trained models and testing data
* 2024/1/16: First release (inference code)

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

Please contact _feifan_eecs@pku.edu.cn_ or open an issue for any questions or
suggestions.

Thanks! :smiley:
