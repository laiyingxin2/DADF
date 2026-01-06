````
# Detect Any Deepfakes (DADF)
**Detect Any Deepfakes: Segment Anything Meets Face Forgery Detection and Localization**

- **Paper (arXiv PDF):** https://arxiv.org/pdf/2306.17075  
- **Base code:** [SAM-Adapter-PyTorch](https://github.com/tianrun-chen/SAM-Adapter-PyTorch)

This repo implements **DADF**, a SAM-based framework for **deepfake detection** and **pixel-level forgery localization**.  
The training/testing pipeline is also reusable for **other segmentation-style tasks** by switching datasets + configs.

---

## Environment

```bash
pip install -r requirements.txt
````

---

## Download SAM Weights

Put checkpoints into `./pretrained/` (or set `sam_checkpoint` in yaml):

* **`default` / `vit_h`**: [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
* **`vit_l`**: [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
* **`vit_b`**: [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

Example:

```bash
mkdir -p pretrained
wget -O pretrained/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

---

## Dataset

### Face Forgery / Deepfake

* **FaceForensics++**: [https://github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)
* **Detect and Locate**: [https://github.com/ChenqiKONG/Detect_and_Locate](https://github.com/ChenqiKONG/Detect_and_Locate)

### Camouflaged Object Detection

* **COD10K**: [https://github.com/DengPingFan/SINet/](https://github.com/DengPingFan/SINet/)
* **CAMO**: [https://drive.google.com/open?id=1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6](https://drive.google.com/open?id=1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6)
* **CHAMELEON**: [https://www.polsl.pl/rau6/datasets/](https://www.polsl.pl/rau6/datasets/)

### Shadow Detection

* **ISTD**: [https://github.com/DeepInsight-PCALab/ST-CGAN](https://github.com/DeepInsight-PCALab/ST-CGAN)

### Polyp Segmentation (Medical)

* **Kvasir**: [https://datasets.simula.no/kvasir-seg/](https://datasets.simula.no/kvasir-seg/)

---

## Download from Google Drive (Method A)

Install:

```bash
pip install -U gdown
```

Example (CAMO):

```bash
mkdir -p load/CAMO
cd load/CAMO

gdown --id 1pRbZVVWRbS3Czqmr7kaQqaSGCiOfEMNr -O CAMO.zip
# or:
# gdown --fuzzy "https://drive.google.com/file/d/1pRbZVVWRbS3Czqmr7kaQqaSGCiOfEMNr/view?usp=drive_link" -O CAMO.zip

unzip -q CAMO.zip
```

---

## Training

Use the provided script:

```bash
bash run.sh
```

Your `run.sh` example:

```bash
CUDA_VISIBLE_DEVICES=0,1,4,3 \
python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --master_port=29501 \
train.py --config configs/demo.yaml
```

---

## Test

```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH]
```

---

## Troubleshooting

### MMCV Version Error

Fix:

```bash
pip uninstall -y mmcv mmcv-full mmcv-lite
pip install -U "mmcv-full==1.7.0" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
```

## Citation

```bibtex
@article{lai2023detect,
  title={Detect Any Deepfakes: Segment Anything Meets Face Forgery Detection and Localization},
  author={Lai, Yingxin and Luo, Zhiming and Yu, Zitong},
  journal={Chinese Conference on Biometric Recognition，CCBR},
  year={2023}
}
```
 
