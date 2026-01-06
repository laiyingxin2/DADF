# Detect Any Deepfakes: Segment Anything Meets Face Forgery Detection and Localization

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2306.17075-red)](https://arxiv.org/pdf/2306.17075)
[![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/laiyingxin2/DADF)

This repository provides the official implementation of **DADF (Detect Any Deepfakes)**, a framework that introduces **Segment Anything Model (SAM)** into **face forgery detection and localization**.

In addition to deepfake tasks, the codebase is also **compatible with other dense prediction tasks** (e.g., camouflaged object detection, shadow detection, polyp segmentation) by switching dataset configs and evaluation metrics.

> This code is built upon the excellent project: **[SAM-Adapter-PyTorch](https://github.com/tianrun-chen/SAM-Adapter-PyTorch)**.

---

## Highlights

- **SAM-based** end-to-end **forgery localization + detection**
- **Multiscale Adapter** for short-/long-range context modeling
- **Reconstruction Guided Attention (RGA)** for improving sensitivity to subtle forged traces
- Can be extended to other segmentation-style tasks with minimal changes

---

## Environment

Recommended environment (aligned with upstream repos):
- Python >= 3.8
- PyTorch (CUDA enabled)

Install dependencies:
```bash
pip install -r requirements.txt
````

---

## Download SAM Weights

Please download SAM checkpoints and place them into `./pretrained/` (or the path specified in your config, e.g. `sam_checkpoint`):

* **`default` or `vit_h`**: [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
* **`vit_l`**: [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
* **`vit_b`**: [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

Example:

```bash
mkdir -p pretrained
# then download a .pth into ./pretrained/
```

> If your config points to another path, follow the config.

---

## Dataset

### Face Forgery (Deepfake) Data Sources

* **[FaceForensics++](https://github.com/ondyari/FaceForensics)**
* **[Detect and Locate](https://github.com/ChenqiKONG/Detect_and_Locate)**

### Camouflaged Object Detection

* **[COD10K](https://github.com/DengPingFan/SINet/)**
* **[CAMO](https://drive.google.com/open?id=1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6)**
* **[CHAMELEON](https://www.polsl.pl/rau6/datasets/)**

### Shadow Detection

* **[ISTD](https://github.com/DeepInsight-PCALab/ST-CGAN)**

### Polyp Segmentation (Medical)

* **[Kvasir](https://datasets.simula.no/kvasir-seg/)**

> Put your processed dataset under `./load/` (or follow the paths defined in `configs/*.yaml`).

---

## Quick Start

### 1) Train (Distributed)

We provide a simple launcher script `run.sh`.

**run.sh example:**

```bash
CUDA_VISIBLE_DEVICES=0,1,4,3 \
python -m torch.distributed.launch \
  --nnodes=1 --nproc_per_node=4 --master_port=29501 \
  train.py --config configs/demo.yaml
```

Or run directly:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
  --nnodes 1 --nproc_per_node 4 --master_port=29501 \
  train.py --config configs/demo.yaml
```

> **Common issue: `EADDRINUSE` / port 29500 already in use**
> Set another port, e.g. `--master_port=29501` / `29511` / `29600`.

---

### 2) Test / Evaluation

```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH]
```

Example:

```bash
python test.py --config configs/demo.yaml --model save/_demo/model_epoch_best.pth
```

---

## Notes

* SAM backbones are memory-hungry. Multi-GPU training is recommended.
* To adapt to other tasks (COD/Shadow/Polyp), switch:

  * dataset config (`configs/*.yaml`)
  * evaluation metric type (`eval_type` in config)
  * dataset wrapper / loader definitions if needed

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{lai2023detect,
  title={Detect Any Deepfakes: Segment Anything Meets Face Forgery Detection and Localization},
  author={Lai, Yingxin and Luo, Zhiming and Yu, Zitong},
  journal={Chinese Conference on Biometric Recognition，CCBR},
  year={2023}
}
```

Paper (arXiv): [https://arxiv.org/pdf/2306.17075](https://arxiv.org/pdf/2306.17075)

---

## Reference

* SAM-Adapter-PyTorch: [https://github.com/tianrun-chen/SAM-Adapter-PyTorch](https://github.com/tianrun-chen/SAM-Adapter-PyTorch)

 
