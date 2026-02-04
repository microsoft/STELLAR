# **STELLAR**

[[`Paper`](https://arxiv.org/abs/2602.01905)] [[`BibTeX`](#Citation)]

This repository hosts the code and resources for **STELLAR**, aka "Learning Sparse Visual Representations via Spatial-Semantic Factorization". **STELLAR** is produce unified sparse vision representation supporting both **reconstruction** (2.60 FID) and **semantics** (79.10% linear probing accuracy), with 90% reduction in the latent size compared to dense grid (only 16 tokens). By factorizing "what" from "where", STELLAR effectively models the multiple semantic concepts in an image along with their spatial localization, enabling efficient, holistic vision representation.


## Installation
```sh
git clone https://github.com/microsoft/STELLAR.git
```

### Conda Environment Setup
```sh
conda create -n stellar python=3.10.14
conda activate stellar
```

Install dependencies
```sh
pip install -r requirements.txt 

pip install azureml-automl-core
pip install opencv-python
```

## Model Weights
Pretrained odel weights will be made availabe soon.

## Model Training

```bash
python -m azureml.acft.image.components.olympus.app.main \
  --config-path <YOUR ABSOLUTE CONFIG DIRECTORY PATH> \
  --config-name stellar
```


## Citation

If you find our paper/code interesting and helpful for your research, please consider citing:

```bibtex
@article{zhao2026stellar,
      title={Learning Sparse Visual Representations via Spatial-Semantic Factorization}, 
      author={Theodore Zhengde Zhao and Sid Kiblawi and Jianwei Yang and Naoto Usuyama and Reuben Tan and Noel C Codella and Tristan Naumann and Hoifung Poon and Mu Wei},
      year={2026},
      eprint={2602.01905},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.01905}, 
}
```

## Usage and License Notices
The model described in this repository is provided for research and development use only. 