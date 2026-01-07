# APDM
Official implementation for NeurIPS 2025 paper:

**Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models**

[Tae-Young Lee](https://github.com/TY-LEE-KR)\*, [Juwon Seo](https://github.com/JJuOn)\*, Jong Hwan Ko<sup>$\dagger$</sup>, and Gyeong-Moon Park<sup>$\dagger$</sup> 

[![arXiv](https://img.shields.io/badge/arXiv-2511.01307-b31b1b.svg)](https://arxiv.org/abs/2511.01307) 

# Environment
- Python 3.12.x
- Torch 2.3.1
- NVIDIA GeForce RTX A6000 (for protection) and 3090 / A5000 (for evaluation)
- CUDA 12.1

# Getting Started
## Environment Setup
```bash
git clone git@github.com/KU-VGI/APDM.git
cd APDM
conda create -n APDM python=3.12
conda activate APDM
pip install -r requirements.txt
```

## Models
Pretrained checkpoints of different Stable Diffusion versions can be **downloaded** from provided links in the table below:
<table style="width:100%">
  <tr>
    <th>Version</th>
    <th>Link</th>
  </tr>
  <tr>
    <td>2.1</td>
    <td><a href="https://huggingface.co/stabilityai/stable-diffusion-2-1-base">stable-diffusion-2-1-base</a></td>
  </tr>
  <tr>
    <td>1.5</td>
    <td><a href="https://huggingface.co/runwayml/stable-diffusion-v1-5">stable-diffusion-v1-5</a></td>
  </tr>
</table>

Please put them in `./models/`. Note: Stable Diffusion version 1.5 is the default version in all of our experiments.

## Additional Files
For easier and reproducible FID evaluation, we provide a .npy file containing pre-extracted COCO feature embeddings. \
You can download the file at [Link](https://drive.google.com/file/d/1YNJOvJnAbe647J3yE31k7lmHRpm3uGbx/view?usp=drive_link).

# How to run
## Data Preparation
We provide sample target data (./data/) and paired data (./paired_set/) to run our code.
- **Person Data** can be downloaded from the [Anti-DreamBooth project](https://github.com/VinAIResearch/Anti-DreamBooth?tab=readme-ov-file#dataset-preparation).
- **Object Data** can be downloaded from the original [DreamBooth repository](https://github.com/google/dreambooth).

## Quick Start
If you want to run the full pipeline with a single command, use:
```bash
source run.sh apdm001   # apdm001–apdm008
```
- apdm001–apdm004: APDM protection for person subjects.
- apdm005–apdm008: APDM protection for dog subjects.

For detailed argument settings, please refer to **gen_arguments.py**.

## Code Structure
- **protect.py**:
Performs the APDM protection process.
- **train_dreambooth.py**:
Conducts personalization based on HuggingFace Diffusers’ DreamBooth.
- **evaluate_db.py**:
Evaluates personalization-time protection performance
(DINO, BRISQUE).
- **evaluate.py**:
Evaluates general generation quality
(FID, CLIP Score).

# Acknowledgement
This project references resources from the following repositories [Dreambooth-Huggingface](https://huggingface.co/docs/diffusers/training/dreambooth), [DreamBooth](https://github.com/google/dreambooth), and [Anti-DreamBooth](https://github.com/VinAIResearch/Anti-DreamBooth?tab=readme-ov-file#dataset-preparation). We thank the authors for releasing their code and datasets publicly.

# BibTex
```
@inproceedings{lee2025perturb,
  title={Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models},
  author={Tae-Young Lee and Juwon Seo and Jong Hwan Ko and Gyeong-Moon Park},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year={2025},
  url={https://openreview.net/forum?id=5XoqKCmkS7}
}
```
