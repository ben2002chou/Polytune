# Polytune

**Detecting Music Performance Errors with Transformers**  
*Accepted at AAAI 2025*

Polytune introduces a transformer-based model for end-to-end music performance error detection.

**[Read the Paper](https://arxiv.org/pdf/2501.02030)**



## Project Overview

![Model Diagram](poster_AAAI.png)



## Demo Video

<p align="center">
  <a href="https://youtu.be/y91Qts1TWBY">
    <img src="https://img.youtube.com/vi/y91Qts1TWBY/0.jpg" alt="Demo Video" width="640"/>
  </a>
</p>

<p align="center">
  <em>Click the thumbnail to watch the demo on YouTube.</em>
</p>



## Environment Setup

1. Install **Python 3.11**
2. Create and activate the Conda environment:

   ```bash
   conda env create -n polytune python=3.11
   conda activate polytune
   ```

3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```



## Running the Code

### Training

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

### Evaluation

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

### Inference

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



## Output Format

The model produces a multi-track MIDI file:
- **Track 1**: Extra notes
- **Track 2**: Missing notes
- **Track 3**: Correct notes



## Datasets

- **CocoChorales-E**: [Download](https://app.globus.org/file-manager?origin_id=55272af5-6d96-4381-a3ea-8d7197e57b33&origin_path=%2F)
- **MAESTRO-E**: [Download](https://app.globus.org/file-manager?origin_id=63862b12-9f93-4f7c-ad7b-5a14ab5d1af2&origin_path=%2F)

Download requires [Globus](https://www.globus.org) or [Globus Connect Personal](https://www.globus.org/globus-connect-personal).

To download:
1. Log in and follow setup instructions  
2. Visit the dataset link  
3. Click **"Transfer or Sync to..."**  
4. Select your local machine or HPC endpoint

You can also generate your own error datasets using our code:  
[CocoChorales-E_MAESTRO-E GitHub Repo](https://github.com/ben2002chou/CocoChorales-E_MAESTRO-E)



## Pretrained Models

Pretrained checkpoints are available here:  
[Download from Hugging Face](https://huggingface.co/ben2002chou/Polytune/tree/main)



## Reproducing Results

The baseline implementation (MT3) can be found in:  
`./baseline/MT3_baseline/`



## Notes

There are a few unused label files currently included in the CocoChorales dataset.  
These are harmless — the data loader automatically filters them.



## Citation

If you use Polytune or our datasets, please cite:

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
```
