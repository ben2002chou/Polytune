import os

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import torch
import pytorch_lightning as pl
import os

import hydra
from tasks.polytune_net import polytune


@hydra.main(config_path="config", config_name="config")
# def main(config, model_config, result_dir, mode, path):
def main(cfg):
    # set seed to ensure reproducibility
    pl.seed_everything(cfg.seed)

    model = hydra.utils.instantiate(cfg.model, optim_cfg=cfg.optim)
    logger = WandbLogger(project=f"{cfg.model_type}_{cfg.dataset_type}")

    # sanity check to make sure the correct model is used
    assert cfg.model_type == cfg.model._target_.split(".")[-1]

    lr_monitor = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(**cfg.modelcheckpoint)
    tqdm_callback = TQDMProgressBar(refresh_rate=1)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback, tqdm_callback],
        **cfg.trainer,
    )

    train_loader = DataLoader(
        hydra.utils.instantiate(cfg.dataset.train),
        **cfg.dataloader.train,
        collate_fn=hydra.utils.get_method(cfg.dataset.collate_fn),
    )

    val_loader = DataLoader(
        hydra.utils.instantiate(cfg.dataset.val),
        **cfg.dataloader.val,
        collate_fn=hydra.utils.get_method(cfg.dataset.collate_fn),
    )

    if cfg.path is not None and cfg.path != "":
        if cfg.path.endswith(".ckpt"):
            print(f"Validating on {cfg.path}...")
            trainer.validate(model, val_loader, ckpt_path=cfg.path)
            print("Training start...")
            trainer.fit(model, train_loader, val_loader, ckpt_path=cfg.path)

        elif cfg.path.endswith(".pth"):
            print(f"Loading weights from {cfg.path}...")
            checkpoint = torch.load(cfg.path, map_location="cpu")
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                # Assuming the entire checkpoint is the model state dict if 'model_state_dict' key is not found
                state_dict = checkpoint
            
            if not isinstance(model.model, torch.nn.parallel.DistributedDataParallel):
                # Remove 'module.' prefix if model is not DDP but state_dict was saved in DDP mode
                new_state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
            else:
                # If loading into a DDP model, no need to adjust the state_dict
                new_state_dict = state_dict
                del state_dict
                
            updated_state_dict = {}
            for key, value in new_state_dict.items():
                if 'decoder' not in key and 'encoder.' not in key:
                    new_key = 'encoder.' + key
                else:
                    new_key = key
                updated_state_dict[new_key] = value
            del state_dict
            
  
                
            
            
            missing_keys, unexpected_keys = model.model.load_state_dict(updated_state_dict, strict=False)
            print("Now loading MAE pretrained weights from", cfg.path, flush=True)
            print("Missing keys:", missing_keys, flush=True)
            print("Unexpected keys:", unexpected_keys, flush=True)
            
            trainer.validate(
                model,
                val_loader,
            )
            print("Training start...")
            trainer.fit(
                model,
                train_loader,
                val_loader,
            )
            

        else:
            raise ValueError(f"Invalid extension for path: {cfg.path}")

    else:
        trainer.fit(
            model,
            train_loader,
            val_loader,
        )

    # save the model in .pt format
    current_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ckpt_path = os.path.join(
        current_dir,
        f"{cfg.model_type}_{cfg.dataset_type}",
        "version_0/checkpoints/last.ckpt",
    )
    # make sure the ckpt_path exists
    if not os.path.exists(ckpt_path):
        # if not, create the directory
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    model.eval()
    dic = {}
    for key in model.state_dict():
        if "model." in key:
            dic[key.replace("model.", "")] = model.state_dict()[key]
        else:
            dic[key] = model.state_dict()[key]
    torch.save(dic, ckpt_path.replace(".ckpt", ".pt"))
    print(f"Saved model in {ckpt_path.replace('.ckpt', '.pt')}.")


if __name__ == "__main__":

    main()
