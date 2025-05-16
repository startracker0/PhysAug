# PhysAug: A Physical-guided and Frequency-based Data Augmentation for Single-Domain Generalized Object Detection
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/physaug-a-physical-guided-and-frequency-based/robust-object-detection-on-cityscapes-1)](https://paperswithcode.com/sota/robust-object-detection-on-cityscapes-1?p=physaug-a-physical-guided-and-frequency-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/physaug-a-physical-guided-and-frequency-based/robust-object-detection-on-dwd)](https://paperswithcode.com/sota/robust-object-detection-on-dwd?p=physaug-a-physical-guided-and-frequency-based)

This repository contains the official implementation of our AAAI 2025 accepted paper:

**"PhysAug: A Physical-guided and Frequency-based Data Augmentation for Single-Domain Generalized Object Detection"**

[Paper](https://arxiv.org/pdf/2412.11807)

## 🎯 Abstract

PhysAug is a novel data augmentation technique designed for single-domain generalized object detection. By leveraging physical priors and frequency-based operations, PhysAug enhances the robustness of detection models under various challenging conditions, such as low-light or motion blur, while maintaining computational efficiency. Extensive experiments demonstrate the superior performance of PhysAug over existing methods, particularly in adverse real-world scenarios.

## 📜 Highlights

- **Physical-guided Augmentation**: Simulates real-world conditions using physical priors.
- **Frequency-based Feature Simulation**: Operates in the frequency domain for precise and computationally efficient augmentation.
- **Improved Robustness**: Enhances model performance in challenging conditions like diverse weather.
- **Single-Domain Generalization**: Outperforms traditional methods without requiring domain adaptation techniques.


## 🚀 Installation
```bash
git clone https://github.com/startracker0/PhysAug.git
cd PhysAug

conda create -n physaug python=3.8 -y
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
pip install -v -e .

pip install -r requirements.txt
```
To ensure reproducibility, the detailed environment dependencies are provided in requirements.txt and environment.yaml

## 📊 Reproducing Results

Follow the steps below to reproduce the results reported in our AAAI 2025 paper.

### 1. Prepare the Dataset
Download and prepare the dataset required for the experiments. Update the dataset path in the configuration file.

#### DWD Dataset
You can download the DWD dataset from the following link:
[Download DWD Dataset](https://drive.google.com/drive/folders/1IIUnUrJrvFgPzU8D6KtV0CXa8k1eBV9B)

#### Cityscapes-C Dataset
The Cityscapes dataset can be downloaded from the official website:
[Download Cityscapes Dataset](https://www.cityscapes-dataset.com/)

We generate the Cityscapes-C validation set based on the cityscapes/leftImg8bit/val portion of the dataset.
You can create this dataset using the [imagecorruptions](https://github.com/bethgelab/imagecorruptions) library, which provides various corruption functions to simulate adverse conditions such as noise, blur, weather, and digital artifacts.

```bash
git clone https://github.com/bethgelab/imagecorruptions.git
cd imagecorruptions
pip install -v -e .
python gen_cityscapes_c.py
```

The datasets should be organized as follows:
```bash
datasets/
├── DWD/
│   ├── daytime_clear/
│   ├── daytime_foggy/
│   ├── dusk_rainy/
│   ├── night_rainy/
│   └── night_sunny/
├── Cityscapes-c/
│   ├── brightness/
│   ├── contrast/
│   ├── defocus_blur/
........
│   └── zoom_blur/
```

### 2. Training the Model

To train the model using PhysAug, follow these steps:

1. Ensure the dataset paths are correctly configured in `configs/_base_/datasets/dwd.py` and `configs/_base_/datasets/cityscapes_detection.py`.
2. Run the following command to start training:

```bash
bash train_dwd.sh
bash train_cityscapes_c.sh
```

### 3. Evaluating the Model

To evaluate the trained model, follow these steps:

1. Specify the dataset to evaluate (e.g., DWD, Cityscapes, or Cityscapes-C).
2. Run the evaluation script with the following command:

```bash
bash test.sh
```

### 4. Pre-trained Models

You can download the pre-trained models, including Physaug_DWD and Physaug_Cityscapes, from [BaiduDisk](https://pan.baidu.com/s/1bSoP0b2Ce4W4_14wwTyxcQ?pwd=6ske) or [GoogleDisk](https://drive.google.com/file/d/1yYE8GrBsJqimCtzLC7kilc2ITivxpAxk/view?usp=drive_link).

If the link is no longer accessible, please feel free to contact me.


