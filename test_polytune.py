import os
import torch
from inference_error import InferenceHandler
from glob import glob
from tqdm import tqdm
import librosa
import hydra
import numpy as np
import json

from evaluate_errors import evaluate_main
from tasks.polytune_net import polytune


#//////////////XXXXXXXX//////////////////
def get_scores(
    model,
    mistakes_audio_dir=None,
    scores_audio_dir=None,
    mel_norm=True,
    eval_dataset="MAESTRO",
    exp_tag_name="test_midis",
    ground_truth=None,
    verbose=True,
    contiguous_inference=False,
    batch_size=1,
    max_length=1024,
):
    handler = InferenceHandler(
        model=model,
        device=torch.device("cuda"),
        mel_norm=mel_norm,
        contiguous_inference=contiguous_inference,
    )

    def func(fname):
        audio, _ = librosa.load(fname, sr=16000)
        print(f"audio_len in seconds: {len(audio)/16000}")
        return audio
    
    if verbose:
        print("Total mistake audio files:", len(mistakes_audio_dir))
        print("Total score audio files:", len(scores_audio_dir))


    print(f"batch_size: {batch_size}")

    for mistake_file, score_file in tqdm(zip(mistakes_audio_dir, scores_audio_dir), total=len(mistakes_audio_dir)):
        # Process each file pair here
        print("Processing:", mistake_file, "and", score_file)
        mistake_audio = func(mistake_file)
        score_audio = func(score_file)

        fname = mistake_file.split("/")[-2]
        outpath = os.path.join(exp_tag_name, fname, "mix.mid") 

        handler.inference(
            mistake_audio=mistake_audio,
            score_audio=score_audio,
            audio_path=fname,
            outpath=outpath,
            batch_size=batch_size,
            max_length=max_length,
            verbose=verbose,
        )
#//////////////XXXXXXXX//////////////////
    if verbose:
        print("Evaluating...")
    current_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    scores, mean_scores = evaluate_main(
        dataset_name=eval_dataset,
        test_midi_dir=os.path.join(current_dir, exp_tag_name), # TODO: this needs to be split by mistake? No. we can split by instrument later
        ground_truth=ground_truth,
    )

    if verbose:
        for key in sorted(list(mean_scores)):
            print("{}: {:.4}".format(key, mean_scores[key]))

    return scores, mean_scores

def _load_MAESTRO_split_info(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        midi_filename_to_number = {
            os.path.basename(path).replace(".midi", ""): str(number)
            for number, path in data["midi_filename"].items()
        }
        split_to_numbers = {split: set() for split in set(data["split"].values())}
        for number, split in data["split"].items():
            split_to_numbers[split].add(str(number))
        return midi_filename_to_number, split_to_numbers

def _build_dataset(root_dir, json_path, split):
    # Load the mapping and splits
    midi_filename_to_number, split_to_numbers = _load_MAESTRO_split_info(
        json_path
    )
    desired_file_numbers = split_to_numbers[split]

    df = []
    mistakes_audio_dir = []
    scores_audio_dir = []
    # Patterns for file discovery
    extra_notes_pattern = os.path.join(
        root_dir, "label", "extra_notes", "**", "*.mid"
    )
    removed_notes_pattern = os.path.join(
        root_dir, "label", "removed_notes", "**", "*.mid"
    )
    correct_notes_pattern = os.path.join(
        root_dir, "label", "correct_notes", "**", "*.mid"
    )
    
    mistake_pattern = os.path.join(root_dir, "mistake", "**", "mix.*")
    score_pattern = os.path.join(root_dir, "score", "**", "mix.*")

    # Find all file paths using the glob patterns and parse identifiers
    extra_notes_files = {
        os.path.normpath(f).split(os.sep)[-3]: f
        for f in glob(extra_notes_pattern, recursive=True)
    }
    removed_notes_files = {
        os.path.normpath(f).split(os.sep)[-3]: f
        for f in glob(removed_notes_pattern, recursive=True)
    }
    correct_notes_files = {
        os.path.normpath(f).split(os.sep)[-3]: f
        for f in glob(correct_notes_pattern, recursive=True)
    }
    mistake_files = {
        os.path.normpath(f).split(os.sep)[-2]: f
        for f in glob(mistake_pattern, recursive=True)
    }
    
    score_files = {
        os.path.normpath(f).split(os.sep)[-2]: f
        for f in glob(score_pattern, recursive=True)
    }

    # Match files based on the common identifier
    for track_id in extra_notes_files.keys():

        file_number = midi_filename_to_number.get(track_id)
        if (
            file_number in desired_file_numbers
            and track_id in removed_notes_files
            and track_id in correct_notes_files
            and track_id in mistake_files
            and track_id in score_files
        ):
            df.append(
                {
                    "track_id": track_id,
                    "extra_notes_midi": extra_notes_files[track_id],
                    "removed_notes_midi": removed_notes_files[track_id],
                    "correct_notes_midi": correct_notes_files[track_id],
                    # TODO: need to use stems_audio
                    "mistake_audio": mistake_files[track_id].replace(
                        ".mid", ".wav"
                    ),
                    "score_audio": score_files[track_id].replace(".mid", ".wav"),
                }
            )
            mistakes_audio_dir.append(mistake_files[track_id].replace(".mid", ".wav"))
            scores_audio_dir.append(score_files[track_id].replace(".mid", ".wav"))
            
            
            
    assert len(df) > 0, "No matching files found. Check the dataset directory."
    
    return df, mistakes_audio_dir, scores_audio_dir

    


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg):
    assert cfg.path
    assert (
        cfg.path.endswith(".pt")
        or cfg.path.endswith("pth")
        or cfg.path.endswith("ckpt")
    ), "Only .pt, .pth, .ckpt files are supported."
    assert cfg.eval.exp_tag_name
    
    dataset, mistakes_audio_dir, scores_audio_dir = _build_dataset(
        root_dir=cfg.dataset.test.root_dir,
        json_path=cfg.dataset.test.split_json_path,
        split="test",
    )

    pl = hydra.utils.instantiate(cfg.model, optim_cfg=cfg.optim)
    print(f"Loading weights from: {cfg.path}")
    if cfg.path.endswith(".ckpt"):
        # load lightning module from checkpoint
        model_cls = hydra.utils.get_class(cfg.model._target_)
        print("torch.cuda.device_count():", torch.cuda.device_count())
        # ///// fix errors /////
        pl = model_cls.load_from_checkpoint(
            cfg.path,
            config=cfg.model.config,
            optim_cfg=cfg.optim,
        )
        model = pl.model
    else:
        # load weights for nn.Module
        model = pl.model
        if cfg.eval.load_weights_strict is not None:
            model.load_state_dict(
                torch.load(cfg.path), strict=cfg.eval.load_weights_strict
            )
        else:
            model.load_state_dict(torch.load(cfg.path), strict=False)

    model.eval()
    # if torch.cuda.is_available():
    #     model.cuda()

    
    
    if cfg.eval.eval_first_n_examples:
        random_offset = 0
        if len(mistakes_audio_dir) > cfg.eval.eval_first_n_examples:
            random_offset = np.random.randint(0, len(mistakes_audio_dir) - cfg.eval.eval_first_n_examples)
        mistakes_audio_dir = mistakes_audio_dir[random_offset: random_offset + cfg.eval.eval_first_n_examples]
        scores_audio_dir = scores_audio_dir[random_offset: random_offset + cfg.eval.eval_first_n_examples]
        # mistakes_audio_dir = mistakes_audio_dir[: cfg.eval.eval_first_n_examples]
        # scores_audio_dir = scores_audio_dir[: cfg.eval.eval_first_n_examples]
    mistakes_audio_dir = ['/depot/yunglu/data/datasets_ben/maestro/maestro_with_mistakes_unaligned/mistake/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_02_R1_2015_wav--1/mix.wav']
    scores_audio_dir = ['/depot/yunglu/data/datasets_ben/maestro/maestro_with_mistakes_unaligned/score/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_02_R1_2015_wav--1/mix.wav']
        

    mel_norm = True
    

    get_scores(
        model,
        mistakes_audio_dir=mistakes_audio_dir,
        scores_audio_dir=scores_audio_dir,
        mel_norm=mel_norm,
        eval_dataset=cfg.eval.eval_dataset,
        exp_tag_name=cfg.eval.exp_tag_name,
        ground_truth=dataset,
        contiguous_inference=cfg.eval.contiguous_inference,
        batch_size=cfg.eval.batch_size,
    )


if __name__ == "__main__":
    main()
