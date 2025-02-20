import os
import json
import torch
from inference import InferenceHandler
from glob import glob
from tqdm import tqdm
import librosa
import hydra
import numpy as np
import json
import pretty_midi

from evaluate_errors_coco_old import evaluate_main
from tasks.polytune_net import polytune

def select_files(audio_dir, n_samples=None):
    """Select a subset of files based on a specified limit."""
    if n_samples is not None and len(audio_dir) > n_samples:
        random_offset = np.random.randint(0, len(audio_dir) - n_samples)
        selected_files = audio_dir[random_offset: random_offset + n_samples]
    else:
        selected_files = audio_dir
    return selected_files

#//////////////XXXXXXXX//////////////////
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
    output_json_path=None,
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
        
        fname = audio_file.split("/")[-3]
        base_name, ext = os.path.splitext(os.path.basename(audio_file))
        file_name = base_name + ".mid"
        print(f"fname: {fname}, file_name: {file_name}", flush=True)
        outpath = os.path.join(exp_tag_name, fname, file_name)

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

    # Evaluating the main function and getting the scores
    mean_scores, mean_track_instrument_scores, track_instrument_scores = evaluate_main(
        dataset_name=eval_dataset,
        test_midi_dir=os.path.join(current_dir, exp_tag_name),  # TODO: this needs to be split by mistake? No. we can split by instrument later
        ground_truth=ground_truth,
        output_json_file=output_json_path,
    )
    
    # Print mean_scores if verbose
    if verbose:
        print("Mean Scores:")
        for key in sorted(mean_scores):
            print("{}: {:.4}".format(key, mean_scores[key]))

        print("\nMean Track and Instrument Scores:")
        for track_index, instruments in sorted(mean_track_instrument_scores.items()):
            print("Track {}: ".format(track_index))
            for instrument, metrics in instruments.items():
                print("  Instrument: {}".format(instrument))
                for metric, score in metrics.items():
                    print("    {}: {:.4}".format(metric, score))

    return mean_scores, mean_track_instrument_scores, track_instrument_scores

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
    
def capitalize_instrument_name(file_path):
    # Extract the base path and file name
    base_path, file_name = os.path.split(file_path)
    # Split the file name into parts before and after the first underscore
    parts = file_name.split('_', 1)  # Only split on the first underscore
    if len(parts) > 1:
        # Split the instrument name part into individual words
        instrument_name_parts = parts[1].split('_')
        # Capitalize each part of the instrument name
        capitalized_instrument_name_parts = [part.capitalize() for part in instrument_name_parts]
        # Join the capitalized parts back with spaces
        capitalized_instrument_name = ' '.join(capitalized_instrument_name_parts)
        # Replace the original instrument part with the capitalized version
        parts[1] = capitalized_instrument_name
        # Reconstruct the file name
        file_name = '_'.join(parts) #.replace('_', ' ', 1)  # Replace only the first underscore with a space
        # Ensure only one .wav extension
        file_name = file_name.replace('.wav.wav', '.wav')
    # Reconstruct the full path
    return os.path.join(base_path, file_name)


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


def _build_dataset(root_dir, json_path, split, output_json_file):
    # Load the mapping and splits
    midi_filename_to_number, split_to_numbers = _load_MAESTRO_split_info(json_path)
    print(f"Finished loading {json_path}", flush=True)
    desired_file_numbers = split_to_numbers[split]

    df = []
    mistakes_audio_dir = []
    scores_audio_dir = []

    # Patterns for file discovery
    extra_notes_dir = os.path.join(root_dir, "label", "extra_notes")
    removed_notes_dir = os.path.join(root_dir, "label", "removed_notes")
    correct_notes_dir = os.path.join(root_dir, "label", "correct_notes")
    mistake_dir = os.path.join(root_dir, "mistake")
    score_dir = os.path.join(root_dir, "score")

    print(f"Finished loading {root_dir}", flush=True)

    # Define directory mapping
    directories = {
        "extra_notes": extra_notes_dir,
        "removed_notes": removed_notes_dir,
        "correct_notes": correct_notes_dir,
        "mistake": mistake_dir,
        "score": score_dir
    }

    def scan_and_save_paths(directories, output_json_file, batch_size=1000):
        files_dict = {key: {} for key in directories}
        batch = []
        batch_count = 0
        total_files = 0
        for key, dir_path in directories.items():
            for root, _, filenames in os.walk(dir_path):
                for filename in filenames:
                    if filename.endswith(".mid") or filename.endswith(".wav"):
                        file_path = os.path.join(root, filename)
                        # Determine the appropriate directory key based on file type
                        dir_key = os.path.normpath(file_path).split(os.sep)[-3] if key in ["extra_notes", "removed_notes", "correct_notes"] else os.path.normpath(file_path).split(os.sep)[-2]
                        if dir_key not in files_dict[key]:
                            files_dict[key][dir_key] = []
                        files_dict[key][dir_key].append(file_path)
                        batch.append(file_path)

                    if len(batch) >= batch_size:
                        batch_count += 1
                        total_files += len(batch)
                        print(f"Processed batch {batch_count} with {len(batch)} files (Total: {total_files})", flush=True)
                        batch.clear()

        if batch:  # Process remaining files in the last batch
            batch_count += 1
            total_files += len(batch)
            print(f"Processed batch {batch_count} with {len(batch)} files (Total: {total_files})", flush=True)

        with open(output_json_file, 'w') as json_file:
            json.dump(files_dict, json_file)
        print(f"File paths saved to {output_json_file}", flush=True)
        return files_dict

    def load_paths_from_json(output_json_file):
        with open(output_json_file, 'r') as json_file:
            files_dict = json.load(json_file)
        print(f"File paths loaded from {output_json_file}", flush=True)
        return files_dict

    # Load or scan file paths
    if os.path.exists(output_json_file):
        print(f"Loading file paths from {output_json_file}", flush=True)
        files_dict = load_paths_from_json(output_json_file)
    else:
        print(f"Scanning and saving file paths to {output_json_file}", flush=True)
        files_dict = scan_and_save_paths(directories, output_json_file)

    extra_notes_files = files_dict["extra_notes"]
    removed_notes_files = files_dict["removed_notes"]
    correct_notes_files = files_dict["correct_notes"]
    mistake_files = files_dict["mistake"]
    score_files = files_dict["score"]

    # Match files based on the common identifier
    for track_id in mistake_files.keys():
        file_number = midi_filename_to_number.get(track_id)
        if file_number in desired_file_numbers:
            num_subtracks = len(mistake_files[track_id])
            if track_id in removed_notes_files and track_id in correct_notes_files and track_id in mistake_files and track_id in score_files:
                for i in range(num_subtracks):
                    if (i < len(removed_notes_files[track_id]) and
                        i < len(correct_notes_files[track_id]) and
                        i < len(mistake_files[track_id]) and
                        i < len(score_files[track_id])):
                        print(f"Processing track_id {track_id}, subtrack index {i}")
                        print(f"mistake_files[track_id][i]: {mistake_files[track_id][i]}")
                        print(f"score_files[track_id][i]: {score_files[track_id][i]}")
                        mistake_audio = capitalize_instrument_name(mistake_files[track_id][i].replace("stems_midi", "stems_audio").replace(".mid", ".wav"))
                        score_audio = capitalize_instrument_name(score_files[track_id][i].replace("stems_midi", "stems_audio").replace(".mid", ".wav"))
                        
                        if os.path.exists(mistake_audio) and os.path.exists(score_audio):
                            df.append({
                                "track_id": track_id,
                                "file_number": file_number,
                                "extra_notes_midi": extra_notes_files[track_id][i],
                                "removed_notes_midi": removed_notes_files[track_id][i],
                                "correct_notes_midi": correct_notes_files[track_id][i],
                                "mistake_audio": mistake_audio,
                                "score_audio": score_audio,
                                "score_midi": score_files[track_id][i].replace("stems_audio", "stems_midi").replace(".wav", ".mid"),
                                "aligned_midi": score_files[track_id][i].replace("stems_audio", "stems_midi").replace(".wav", "_aligned.mid"),
                            })
                            mistakes_audio_dir.append(mistake_audio)
                            scores_audio_dir.append(score_audio)
                        else:
                            if not os.path.exists(mistake_audio):
                                print(f"File does not exist: {mistake_audio}")
                            if not os.path.exists(score_audio):
                                print(f"File does not exist: {score_audio}")
                    else:
                        print(f"Index out of range for track_id {track_id}, subtrack index {i}")
            else:
                pass
                # print(f"Missing track data for {track_id}")
        else:
            pass
            # print(f"Track {track_id} not in desired file numbers")

    assert len(df) > 0, "No matching files found. Check the dataset directory."
    print("Total files:", len(df))
    return df, mistakes_audio_dir, scores_audio_dir
####################################################################################################
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
    output_json_path = os.path.join(os.path.dirname(cfg.dataset.test.split_json_path), "file_paths_8_12.json")
    print(f"Output json path: {output_json_path}")
    output_json_path = '/home/chou150/depot/datasets/cocochorales_full/file_paths_8_12.json'

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

        
    dataset, mistakes_audio_dir, scores_audio_dir = _build_dataset(
        root_dir=cfg.dataset.test.root_dir,
        json_path=cfg.dataset.test.split_json_path,
        split="test",
        output_json_file=output_json_path,
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
        output_json_path=output_json_path,
    )


if __name__ == "__main__":
    main()
