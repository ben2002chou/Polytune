# Polytune

**Detecting Music Performance Errors with Transformers**  
*AAAI 2025*

Polytune introduces a transformer-based model for end-to-end music error detection.

**[Read the Paper](https://arxiv.org/pdf/2501.02030)**
## 📊 Project Overview

![Diagram](poster_AAAI.png)

## 🎥 Demo

Click below to watch the demo:

[![Demo Video](https://img.youtube.com/vi/y91Qts1TWBY/0.jpg)](https://youtu.be/y91Qts1TWBY)


## How to Run

Clone the repository and set up the environment:
```bash
git clone <repository-url>
cd Polytune
conda env create -n polytune python==3.XX
conda activate polytune
```

## Environment Setup

Ensure you have the correct environment configuration:
- Install Python 3.XX
- Create and activate the Conda environment:
  ```bash
  conda env create -n polytune python==3.11
  conda activate polytune
  ```
- Install the required dependencies (via pip):
  ```bash
  pip install -r requirements.txt
  ```

Training:
```bash
python train_polytune.py \
  --config-path="config" \
  --config-name="config_maestro/coco" \
  'devices=[0]' \
  'hydra/job_logging=disabled' \
  'model="polytune"' \
  'dataset="MAESTRO/CocoChorales"' \
  'split_frame_length=2000'
```

Evaluation:
```bash
python test_polytune.py \
  --config-dir="config" \
  --config-name="config_maestro/coco" \
  model="polytune" \
  path="pretrained.ckpt" \
  eval.eval_dataset="MAESTRO/CocoChorales" \
  eval.exp_tag_name="Polytune" \
  hydra/job_logging=disabled \
  eval.is_sanity_check=True \
  eval.contiguous_inference=True \
  split_frame_length=2000
```

Inference:
```bash
python polytune_test_inference.py \
  --config-dir="config" \
  --config-name="config_maestro" \
  model="polytune" \
  path="pretrained.ckpt" \
  hydra/job_logging=disabled \
  eval.is_sanity_check=True \
  eval.contiguous_inference=True \
  split_frame_length=2000
```
## Interpeting Outputs

The output is a MIDI file containing three tracks:

- Track 1: Extra notes
- Track 2: Missing notes
- Track 3: Correct notes

## Datasets

- CocoChorales-E: [link](https://app.globus.org/file-manager?origin_id=55272af5-6d96-4381-a3ea-8d7197e57b33&origin_path=%2F)
- MAESTRO-E: [link](https://app.globus.org/file-manager?origin_id=63862b12-9f93-4f7c-ad7b-5a14ab5d1af2&origin_path=%2F)

- Downloading this dataset requires [Globus](https://www.globus.org) or [Globus Connect Personal](https://www.globus.org/globus-connect-personal).

1. Log in and follow setup instructions  
2. Go to the dataset link 
3. Click **“Transfer or Sync to…”**  
4. Choose your machine or HPC endpoint as the destination

You can also generate your own error datasets based on our code [here](https://github.com/ben2002chou/CocoChorales-E_MAESTRO-E):
## Pre-trained models

Pretrained weights can be downloaded here:
- [Download Weights](https://huggingface.co/ben2002chou/Polytune/tree/main)


## Reproducing Results

Baseline: Located in ./baseline/MT3_baseline

## TODO

There are some label files in the current download link for CocoChorales that aren't used. These need to be removed. This doesn't affect the usage of the code as the dataset loading code filters out redundant labels.


## Citation

If you use our dataset in your research, please cite our paper:

```bibtex
@inproceedings{chou_detecting_2025,
  author    = {Chou, Benjamin Shiue-Hal and Jajal, Purvish and Eliopoulos, Nicholas John 
               and Nadolsky, Tim and Yang, Cheng-Yun and Ravi, Nikita and Davis, James C. 
               and Yun, Kristen Yeon-Ji and Lu, Yung-Hsiang},
  title     = {Detecting Music Performance Errors with Transformers},
  booktitle = {AAAI Conference on Artificial Intelligence},
  publisher = {AAAI},
  year      = {2025}
}
