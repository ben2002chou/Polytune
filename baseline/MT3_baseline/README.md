# Baseline
Note: Our implementation of MT3 is fully in Pytorch so the original MT3 weight cannot be applied here. We retrain from scratch using the following:

## How to Run

### Training
```bash
srun python mt3_train[_coco].py \
  --config-path="config" \
  --config-name="config_maestro/coco_old.yaml" \
  'devices=[0]' \
  'hydra/job_logging=disabled' \
  'model="MT3Net"' \
  'dataset="MAESTRO/CocoChorales_old"' \
  'split_frame_length=2000'
```

### Evaluation
```bash
srun python test_mt3[_coco]_old.py \
  --config-dir=config \
  --config-name=config_maestro/coco_old \
  model=MT3Net \
  path='mt3.ckpt' \
  eval.eval_dataset="MAESTRO/CocoChorales_old" \
  eval.exp_tag_name="mt3" \
  hydra/job_logging=disabled \
  eval.is_sanity_check=True \
  eval.contiguous_inference=True \
  split_frame_length=2000 \
  eval.eval_first_n_examples=5
```
