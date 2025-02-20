import os
import torch
import hydra
from inference_error import InferenceHandler
import librosa
from omegaconf import DictConfig

def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    print(f"Loaded {file_path} with length {len(audio)/sr} seconds.")
    return audio

def run_inference(model, mistake_file, score_file, output_dir, batch_size=1, max_length=1024):
    handler = InferenceHandler(model=model, device=torch.device("cuda"))

    mistake_audio = load_audio(mistake_file)
    score_audio = load_audio(score_file)

    fname = os.path.basename(mistake_file).split(".")[0]
    outpath = os.path.join(output_dir, f"{fname}_output_1.mid")

    handler.inference(
        mistake_audio=mistake_audio,
        score_audio=score_audio,
        audio_path=fname,
        outpath=outpath,
        batch_size=batch_size,
        max_length=max_length,
        verbose=True,
    )
    print(f"Inference completed. Output saved to {outpath}")

@hydra.main(config_path=None, config_name=None, version_base="1.1")
def main(cfg: DictConfig):
    assert cfg.path, "Model path must be specified in the config file"
    
    # Load the model using Hydra config
    pl = hydra.utils.instantiate(cfg.model, optim_cfg=cfg.optim)
    print(f"Loading weights from: {cfg.path}")
    
    if cfg.path.endswith(".ckpt"):
        # Load from a PyTorch Lightning checkpoint
        model_cls = hydra.utils.get_class(cfg.model._target_)
        pl = model_cls.load_from_checkpoint(cfg.path, config=cfg.model.config, optim_cfg=cfg.optim)
        model = pl.model
        
        # Check if the model's state_dict is loaded correctly
        state_dict = pl.state_dict()
        print("Loaded state dict keys:", state_dict.keys())
    else:
        print("Only .ckpt file loading is supported in this script.")
    
    model.eval()
    
    if torch.cuda.is_available():
        model.cuda()
    
    # Directly assign paths for mistake and score files here
    mistake_file = "/home/chou150/code/Muse/physical_test_data/MHLL_mistake.wav"
    score_file = "/home/chou150/code/Muse/physical_test_data/MHLL_score.wav"
    output_dir = "/home/chou150/code/Muse/physical_test_data/"
    
    run_inference(model, mistake_file, score_file, output_dir)

if __name__ == "__main__":
    main()