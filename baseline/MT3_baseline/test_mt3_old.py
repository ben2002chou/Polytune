# Adapted from https://github.com/username/mr-mt3
#
# coding=utf-8
# Copyright 2024 MR-MT3 Authors (Hao Hao Tan, Kin Wai Cheuk, Taemin Cho, Wei-Hsiang Liao, Yuki Mitsufuji)
#
# Licensed under the MIT License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This code is adapted from the MR-MT3 project: 
# "MR-MT3: Memory Retaining Multi-Track Music Transcription to Mitigate Instrument Leakage"
# Original repository: https://github.com/username/mr-mt3
import os
import json
import torch
from inference import InferenceHandler
from glob import glob
import os
from tqdm import tqdm
import librosa
import hydra
import numpy as np
from evaluate_errors_old import evaluate_main
from tasks.mt3_net import MT3Net
import librosa
import pretty_midi

def select_files(audio_dir, n_samples=None):
    """Select a subset of files based on a specified limit."""
    if n_samples is not None and len(audio_dir) > n_samples:
        random_offset = np.random.randint(0, len(audio_dir) - n_samples)
        selected_files = audio_dir[random_offset: random_offset + n_samples]
    else:
        selected_files = audio_dir
    return selected_files

def get_scores(
    model,
    eval_audio_dir=None,
    mel_norm=True,
    eval_dataset="Score_Informed",
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
        print("Total audio files:", len(eval_audio_dir))
        


    print(f"batch_size: {batch_size}")

    for audio_file in tqdm(eval_audio_dir, total=len(eval_audio_dir)):
        # Process each file pair here
        print("Processing:", audio_file)
        audio = func(audio_file)

        fname = audio_file.split("/")[-2]
        outpath = os.path.join(exp_tag_name, fname, "mix.mid") 

        handler.inference(
            audio=audio,
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


def normalize(x):
    """Normalize the values in x."""
    if np.sum(x, axis=0) == 0:
        return np.zeros_like(x)
    return x / np.sum(x, axis=0)

def manual_chroma(piano_roll, start_note=24):
    """Calculate the chroma features manually from the piano roll."""
    num_columns = piano_roll.shape[1]
    note_indices = (np.arange(piano_roll.shape[0]) + start_note) % 12
    chroma = np.zeros((12, num_columns))
    for pitch in range(12):
        chroma[pitch] = np.sum(piano_roll[note_indices == pitch, :], axis=0)
    return chroma

def apply_normalization(arr):
    """Apply normalization across features."""
    x = arr.T
    for i in range(len(x)):
        x[i] = normalize(x[i])
    return x.T

def process_audio_and_midi(audio_path, midi_path):
    """Process audio and MIDI files to align MIDI notes to the audio, skip if output exists."""
    # Modify the output path to include "_aligned"
    output_path = midi_path.replace('.mid', '_aligned.mid')

    # Check if the file already exists
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping processing.")
        return

    # Load audio
    y, sr = librosa.load(audio_path)
    
    # Compute audio chroma and CQT
    FS = 10
    hop_length = int(sr / FS)
    audio_chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    audio_chroma = apply_normalization(audio_chroma)

    # Load MIDI
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    piano_roll = midi_data.get_piano_roll(fs=FS)[24:24+84]
    midi_chroma = apply_normalization(manual_chroma(piano_roll))

    # DTW alignment
    from librosa.sequence import dtw
    _, wp = dtw(audio_chroma, midi_chroma, subseq=True, backtrack=True)

    # Adjust MIDI timings based on DTW path
    adjusted_midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=midi_data.instruments[0].program)
    for note in midi_data.instruments[0].notes:
        midi_frame = int(note.start * FS)
        audio_frame = wp[0][0]
        for i in range(len(wp)):
            if wp[i][1] <= midi_frame:
                audio_frame = wp[min(len(wp)-1, i+1)][0]
                break
        adjusted_start = audio_frame / FS
        adjusted_end = adjusted_start + (note.end - note.start)
        instrument.notes.append(pretty_midi.Note(start=adjusted_start, end=adjusted_end, velocity=note.velocity, pitch=note.pitch))
    adjusted_midi.instruments.append(instrument)

    # Save adjusted MIDI
    adjusted_midi.write(output_path)
    print(f"Processed and saved as {output_path}")
    

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
            and os.path.exists(mistake_files[track_id].replace(".mid", ".wav"))
        ):
            df.append(
                {
                    "track_id": track_id,
                    "extra_notes_midi": extra_notes_files[track_id],
                    "removed_notes_midi": removed_notes_files[track_id],
                    "correct_notes_midi": correct_notes_files[track_id],
                    "mistake_audio": mistake_files[track_id].replace(
                        ".mid", ".wav"
                    ),
                    "score_audio": score_files[track_id].replace(".mid", ".wav"),
                    "score_midi": score_files[track_id].replace(".wav", ".mid"),
                    "aligned_midi": score_files[track_id].replace(".wav", "_aligned.mid"),
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
    
    use_mistake_audio = True
    use_score_audio = False

    pl = hydra.utils.instantiate(cfg.model, optim_cfg=cfg.optim)
    print(f"Loading weights from: {cfg.path}")
    # Loading checkpoint based on the extension
    if cfg.path.endswith(".ckpt"):
        # load lightning module from checkpoint
        model_cls = hydra.utils.get_class(cfg.model._target_)
        print("torch.cuda.device_count():", torch.cuda.device_count())
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

    
        # mistakes_audio_dir = mistakes_audio_dir[: cfg.eval.eval_first_n_examples]
        # scores_audio_dir = scores_audio_dir[: cfg.eval.eval_first_n_examples]

    
    
    dataset, mistakes_audio_dir, scores_audio_dir = _build_dataset(
        root_dir="/home/chou150/depot/datasets/maestro/maestro_with_mistakes_unaligned",
        json_path="/home/chou150/depot/datasets/maestro/maestro-v3.0.0/maestro-v3.0.0.json",
        split="test",
    )
    
    if use_mistake_audio:
        eval_audio_dir = mistakes_audio_dir
    elif use_score_audio:
        eval_audio_dir = scores_audio_dir
    else:
        eval_audio_dir = mistakes_audio_dir + scores_audio_dir
    eval_audio_dir = select_files(eval_audio_dir, cfg.eval.eval_first_n_examples)
    
    for data in dataset:
        if data["mistake_audio"] in eval_audio_dir or data["score_audio"] in eval_audio_dir:
            process_audio_and_midi(data["mistake_audio"], data["score_midi"])
        
    get_scores(
        model,
        eval_audio_dir=eval_audio_dir,
        mel_norm=True,
        eval_dataset=cfg.eval.eval_dataset,
        exp_tag_name=cfg.eval.exp_tag_name,
        ground_truth=dataset,
        contiguous_inference=cfg.eval.contiguous_inference,
        batch_size=cfg.eval.batch_size,
    )


if __name__ == "__main__":
    main()
