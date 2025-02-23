# Polytune

**Detecting Music Performance Errors with Transformers**  
*AAAI 2025*

Polytune introduces a transformer-based model for end-to-end music error detection.

**[Read the Paper](https://arxiv.org/pdf/2501.02030)**

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

## Datasets

- CocoChorales-E: [link](#)
- MAESTRO-E: [link](#)

Note: Actual Links will be added soon.

You can also generate your own error datasets based on our code [here](https://github.com/ben2002chou/CocoChorales-E_MAESTRO-E):
## Pre-trained models

Pretrained weights can be downloaded here:
- [Download Weights](#)

Note: Actual Links will be added soon.

## Reproducing Results

Baseline: Located in ./baseline/MT3_baseline

## Citation
If you use our dataset in your research, please consider citing our paper:
@inproceedings{chou_detecting_2025,
  author = {Chou, Benjamin Shiue-Hal and Jajal, Purvish and Eliopoulos, Nicholas John and Nadolsky, Tim and Yang, Cheng-Yun and Ravi, Nikita and Davis, James C. and Yun, Kristen Yeon-Ji and Lu, Yung-Hsiang},
  booktitle = {{AAAI} {Conference} on {Artificial} {Intelligence}},
  publisher = {AAAI},
  title = {Detecting {Music} {Performance} {Errors} with {Transformers}},
  year = {2025}
}