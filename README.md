<div align="center">

<h1>Detect Any Deepfakes (DADF)</h1>

<h3>Detect Any Deepfakes: Segment Anything Meets Face Forgery Detection and Localization</h3>

<p>
  <b>Yingxin Lai</b><sup>1</sup>, 
  <b>Zhiming Luo</b><sup>1</sup>, 
  <b>Zitong Yu</b><sup>1,2*</sup>
</p>

<p>
  <sup>1</sup> Xiamen University &nbsp;&nbsp; 
  <sup>2</sup> Great Bay University
</p>

<p>
  <a href="https://arxiv.org/pdf/2306.17075">
    <img src="https://img.shields.io/badge/Paper-CCBR%202023-b31b1b.svg?style=flat-square" alt="Paper">
  </a>
  <a href="https://arxiv.org/abs/2306.17075">
    <img src="https://img.shields.io/badge/arXiv-2306.17075-red?style=flat-square" alt="arXiv">
  </a>
  <a href="https://github.com/tianrun-chen/SAM-Adapter-PyTorch">
    <img src="https://img.shields.io/badge/Base-SAM--Adapter-blue?style=flat-square" alt="Base">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
  </a>
</p>

</div>

<br/>

---

## 📖 Introduction

This repository is the official implementation of **[Detect Any Deepfakes](https://arxiv.org/pdf/2306.17075)** (CCBR 2023). 

We propose **DADF**, a novel framework that adapts the **Segment Anything Model (SAM)** for the task of **face forgery detection and localization**. By incorporating low-level forgery traces into the SAM architecture, our model achieves state-of-the-art performance in pixel-level forgery localization while maintaining the versatility to be applied to other segmentation tasks (e.g., camouflage, shadow detection).

## 🛠️ Environment

To ensure reproducibility, we recommend using a clean Conda environment.

### 1. Installation

```bash
# Create environment
conda create -n dadf python=3.8 -y
conda activate dadf

# Install PyTorch (Adjust CUDA version based on your driver)
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url [https://download.pytorch.org/whl/cu117](https://download.pytorch.org/whl/cu117)

# Install Dependencies
pip install -r requirements.txt

```

### 2. Download SAM Weights

Please download the pre-trained weights and put them into the `pretrained/` directory.

* **`default` or `vit_h**`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) (Recommended)
* `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
* `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## 📂 Data Preparation

Please organize your datasets under the `./load/` directory.

| Task | Dataset | Source Link |
| --- | --- | --- |
| **Deepfake** | **FaceForensics++** | [Official Github](https://github.com/ondyari/FaceForensics) |
|  | **Detect & Locate** | [Official Github](https://github.com/ChenqiKONG/Detect_and_Locate) |
| **Camouflage** | **COD10K** | [Official Github](https://github.com/DengPingFan/SINet/) |
|  | **CAMO** | [Google Drive](https://drive.google.com/open?id=1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6) |
| **Shadow** | **ISTD** | [Official Github](https://github.com/DeepInsight-PCALab/ST-CGAN) |
| **Medical** | **Kvasir (Polyp)** | [Official Website](https://datasets.simula.no/kvasir-seg/) |

## ⚡ Quick Start

1. Download the dataset and put it in `./load`.
2. Download the pre-trained [SAM (Segment Anything)](https://github.com/facebookresearch/segment-anything) and put it in `./pretrained`.
3. **Training**:

```bash
# run.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --master_port=29501 \
train.py --config configs/demo.yaml

```

> ⚠️ **Note**: The SAM model consumes significant memory. We use **4 x A100** graphics cards for training. If you encounter memory issues, please try using graphics cards with larger VRAM or reduce the batch size.

## 🚀 Usage

### 1. Training

We support distributed training (DDP). Adjust `nproc_per_node` based on your GPU availability.

```bash
# Example: Train on 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --master_port=29500 \
train.py --config configs/dadf_vit_h.yaml

```

> **Note**: If you encounter OOM (Out of Memory) errors with ViT-H, try reducing the batch size in the config or switching to ViT-B/L backbones.

### 2. Testing

Evaluate the model performance and generate localization maps.

```bash
python test.py --config configs/dadf_vit_h.yaml --model checkpoints/dadf_best.pth

```

## 📝 Citation

If you find this code or paper useful, please cite our work:

```bibtex
@article{lai2023detect,
  title={Detect Any Deepfakes: Segment Anything Meets Face Forgery Detection and Localization},
  author={Lai, Yingxin and Luo, Zhiming and Yu, Zitong},
  journal={Chinese Conference on Biometric Recognition (CCBR)},
  year={2023}
}

```

## 🤝 Acknowledgements

This project is built upon [SAM-Adapter](https://github.com/tianrun-chen/SAM-Adapter-PyTorch). We thank the authors for their excellent codebase.



 
你可以直接复制上面的内容覆盖到你的 `README.md` 文件中。需要我帮你检查 `requirements.txt` 的依赖版本兼容性吗？

```
