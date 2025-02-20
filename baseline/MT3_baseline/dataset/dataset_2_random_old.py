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
#
# Modifications made:
# - Added functionality to handle error classes for Music Error Detection.

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json
import random
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import librosa
import note_seq
from glob import glob
from contrib_old import (
    event_codec,
    note_sequences,
    spectrograms,
    vocabularies,
    run_length_encoding,
    metrics_utils,
)
from contrib_old.preprocessor import (
    slakh_class_to_program_and_is_drum,
    add_track_to_notesequence,
    PitchBendError,
)

MIN_LOG_MEL = -12
MAX_LOG_MEL = 5


class Dataset(Dataset):

    def __init__(
        self,
        root_dir,
        split_json_path,
        split,
        mel_length=256,
        event_length=1024,
        is_train=True,
        include_ties=True,
        ignore_pitch_bends=True,
        onsets_only=False,
        audio_filename="mix.flac",
        midi_folder="MIDI",
        shuffle=True,
        num_rows_per_batch=8,
        split_frame_length=2000,
        is_randomize_tokens=True,
        is_deterministic=False,
        use_mistake=False,
        use_score=True,
    ) -> None:
        super().__init__()
        self.spectrogram_config = spectrograms.SpectrogramConfig()

        self.codec = vocabularies.build_codec(
            vocab_config=vocabularies.VocabularyConfig(num_velocity_bins=1)
        )
        self.vocab = vocabularies.vocabulary_from_codec(self.codec)
        self.audio_filename = audio_filename
        self.midi_folder = midi_folder
        self.mel_length = mel_length
        self.event_length = event_length
        self.df = self._build_dataset(root_dir, split_json_path, split, shuffle=shuffle, train_mistake_audio=use_mistake, train_score_audio=use_score)
        self.is_train = is_train
        self.include_ties = include_ties
        self.ignore_pitch_bends = ignore_pitch_bends
        self.onsets_only = onsets_only
        self.tie_token = (
            self.codec.encode_event(event_codec.Event("tie", 0))
            if self.include_ties
            else None
        )
        self.num_rows_per_batch = num_rows_per_batch
        self.split_frame_length = split_frame_length
        self.is_deterministic = is_deterministic
        self.is_randomize_tokens = is_randomize_tokens

    def _load_MAESTRO_split_info(self, json_path):
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

    def capitalize_instrument_name(self, file_path):
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

    def _build_dataset(self, root_dir, json_path, split, shuffle=True, train_mistake_audio=True, train_score_audio=False):
        # Load the mapping and splits
        midi_filename_to_number, split_to_numbers = self._load_MAESTRO_split_info(
            json_path
        )
        desired_file_numbers = split_to_numbers[split]

        df = []

        mistake_pattern = os.path.join(root_dir, "mistake", "**", "mix.*")
        score_pattern = os.path.join(root_dir, "score", "**", "mix.*")

        mistake_files = {
            os.path.normpath(f).split(os.sep)[-2]: f
            for f in glob(mistake_pattern, recursive=True)
        }
        
        score_files = {
            os.path.normpath(f).split(os.sep)[-2]: f
            for f in glob(score_pattern, recursive=True)
        }
        print(f"score files: {len(score_files)}")
        
        # Match files based on the common identifier
        if train_mistake_audio:
            for track_id in mistake_files.keys():
                file_number = midi_filename_to_number.get(track_id)
                if (
                    file_number in desired_file_numbers
                    and track_id in mistake_files
                ):
                    df.append(
                        {
                            "midi_path": mistake_files[track_id].replace(".wav", ".mid"),
                            "audio_path": mistake_files[track_id].replace(".mid", ".wav"),
                        }
                    )
            
        elif train_score_audio:   
            for track_id in score_files.keys():

                file_number = midi_filename_to_number.get(track_id)
                if (
                    file_number in desired_file_numbers
                    and track_id in score_files
                ):
                    df.append(
                        {
                            "midi_path": score_files[track_id].replace(".wav", ".mid"),
                            "audio_path": score_files[track_id].replace(".mid", ".wav"),
                        }
                    )
        else:
            for track_id in mistake_files.keys():
                file_number = midi_filename_to_number.get(track_id)
                if (
                    file_number in desired_file_numbers
                    and track_id in mistake_files
                    or track_id in score_files
                ):
                    df.append(
                        {
                            "midi_path": mistake_files[track_id].replace(".wav", ".mid"),
                            "audio_path": mistake_files[track_id].replace(".mid", ".wav"),
                        }
                    )
                    df.append(
                        {
                            "midi_path": score_files[track_id].replace(".wav", ".mid"),
                            "audio_path": score_files[track_id].replace(".mid", ".wav"),
                        }
                    )
        assert len(df) > 0, "No matching files found. Check the dataset directory."
        print("total file:", len(df))
        if shuffle:
            random.shuffle(df)

        return df

    def _audio_to_frames(
        self,
        samples: Sequence[float],
    ) -> Tuple[Sequence[Sequence[int]], np.ndarray]:
        """Convert audio samples to non-overlapping frames and frame times."""
        frame_size = self.spectrogram_config.hop_width
        samples = np.pad(
            samples, [0, frame_size - len(samples) % frame_size], mode="constant"
        )

        frames = spectrograms.split_audio(samples, self.spectrogram_config)

        num_frames = len(samples) // frame_size

        times = np.arange(num_frames) / self.spectrogram_config.frames_per_second
        return frames, times


    def _parse_midi(self, path):
        """
        __getitem__ loads the MIDIfiles as a list of note sequences and a list of instrument values.
        Parses MIDI files and returns note sequences and instrument values.

        Args:
            path (str): The path to the directory containing the MIDI files.
            instrument_dict (Dict[str, str]): A dictionary mapping filenames to instrument values.

        Returns:
            Tuple[List[NoteSequence], List[str]]: A tuple containing a list of note sequences and a list of instrument values.
        """
        note_seqs = []
        note_seqs.append(note_seq.midi_file_to_note_sequence(path))
        return note_seqs

    def _tokenize(
        self,
        tracks: List[note_seq.NoteSequence],
        samples: np.ndarray,
        example_id: Optional[str] = None,
    ):

        frames, frame_times = self._audio_to_frames(samples)

        # Add all the notes from the tracks to a single NoteSequence.
        ns = note_seq.NoteSequence(ticks_per_quarter=220)
        # tracks = [note_seq.NoteSequence.FromString(seq) for seq in sequences]
        

        for track in tracks:
            # Instrument name should be Slakh class.
            program = 0
            try:
                add_track_to_notesequence(
                    ns,
                    track,
                    program=program,
                    is_drum=False,
                    ignore_pitch_bends=self.ignore_pitch_bends,
                )
            except PitchBendError:
                # TODO(iansimon): is there a way to count these?
                return

        note_sequences.assign_instruments(ns)
        note_sequences.validate_note_sequence(ns)
        if self.is_train:
            # Trim overlapping notes in training (as our event vocabulary cannot
            # represent them), but preserve original NoteSequence for eval.
            ns = note_sequences.trim_overlapping_notes(ns)

        if example_id is not None:
            ns.id = example_id

        if self.onsets_only:
            times, values = note_sequences.note_sequence_to_onsets(ns)
        else:
            times, values = (
                note_sequences.note_sequence_to_onsets_and_offsets_and_programs(ns)
            )
            # print('times', times)
            # print('values', values)

        (
            events,
            event_start_indices,
            event_end_indices,
            state_events,
            state_event_indices,
        ) = run_length_encoding.encode_and_index_events(
            state=note_sequences.NoteEncodingState() if self.include_ties else None,
            event_times=times,
            event_values=values,
            encode_event_fn=note_sequences.note_event_data_to_events,
            codec=self.codec,
            frame_times=frame_times,
            encoding_state_to_events_fn=(
                note_sequences.note_encoding_state_to_events
                if self.include_ties
                else None
            ),
        )

        # print('events', events)
        # print('inputs', np.array(frames).shape)
        # print('frame_times', frame_times.shape)
        # print('targets', events.shape)
        return {
            "inputs": np.array(frames),
            "input_times": frame_times,
            "targets": events,
            "input_event_start_indices": event_start_indices,
            "input_event_end_indices": event_end_indices,
            "state_events": state_events,
            "input_state_event_indices": state_event_indices,
            # 'sequence': ns.SerializeToString()
        }

    def _extract_target_sequence_with_indices(
        self, features, state_events_end_token=None
    ):
        """Extract target sequence corresponding to audio token segment."""
        target_start_idx = features["input_event_start_indices"][0]
        target_end_idx = features["input_event_end_indices"][-1]

        features["targets"] = features["targets"][target_start_idx:target_end_idx]
        # print('features[targets]', [self.get_token_name(k) for k in features['targets']])

        if state_events_end_token is not None:
            # Extract the state events corresponding to the audio start token, and
            # prepend them to the targets array.
            state_event_start_idx = features["input_state_event_indices"][0]
            state_event_end_idx = state_event_start_idx + 1
            while (
                features["state_events"][state_event_end_idx - 1]
                != state_events_end_token
            ):
                state_event_end_idx += 1
            features["targets"] = np.concatenate(
                [
                    features["state_events"][state_event_start_idx:state_event_end_idx],
                    features["targets"],
                ],
                axis=0,
            )
            # print('features[targets] 2', [self.get_token_name(k) for k in features['targets']])

        return features

    def _run_length_encode_shifts(
        self,
        features,
        state_change_event_types=["velocity", "program"],
        feature_key="targets",
    ):
        state_change_event_ranges = [
            self.codec.event_type_range(event_type)
            for event_type in state_change_event_types
        ]

        events = features[feature_key]

        shift_steps = 0
        total_shift_steps = 0
        output = []

        current_state = np.zeros(len(state_change_event_ranges), dtype=np.int32)
        for j, event in enumerate(events):
            if self.codec.is_shift_event_index(event):
                shift_steps += 1
                total_shift_steps += 1
            else:
                # If this event is a state change and has the same value as the current
                # state, we can skip it entirely.

                # NOTE: this needs to be uncommented if not using random-order augmentation
                # because random-order augmentation use `_remove_redundant_tokens` to replace this part
                if not self.is_randomize_tokens:
                    is_redundant = False
                    for i, (min_index, max_index) in enumerate(
                        state_change_event_ranges
                    ):
                        if (min_index <= event) and (event <= max_index):
                            if current_state[i] == event:
                                is_redundant = True
                            current_state[i] = event
                    if is_redundant:
                        continue

                # Once we've reached a non-shift event, RLE all previous shift events
                # before outputting the non-shift event.
                if shift_steps > 0:
                    shift_steps = total_shift_steps
                    while shift_steps > 0:
                        output_steps = np.minimum(
                            self.codec.max_shift_steps, shift_steps
                        )
                        output = np.concatenate([output, [output_steps]], axis=0)
                        shift_steps -= output_steps
                output = np.concatenate([output, [event]], axis=0)

        features[feature_key] = output
        print("output", output)
        return features

    def _remove_redundant_tokens(
        self,
        events,
        state_change_event_types=["velocity", "program"],
    ):
        state_change_event_ranges = [
            self.codec.event_type_range(event_type)
            for event_type in state_change_event_types
        ]

        output = []

        current_state = np.zeros(len(state_change_event_ranges), dtype=np.int32)
        for j, event in enumerate(events):
            # If this event is a state change and has the same value as the current
            # state, we can skip it entirely.

            is_redundant = False
            for i, (min_index, max_index) in enumerate(state_change_event_ranges):
                if (min_index <= event) and (event <= max_index):
                    if current_state[i] == event:
                        is_redundant = True
                    current_state[i] = event
            if is_redundant:
                continue

            # Once we've reached a non-shift event, RLE all previous shift events
            # before outputting the non-shift event.
            output = np.concatenate([output, [event]], axis=0)

        return output

    def _compute_spectrogram(self, ex):
        samples = spectrograms.flatten_frames(
            ex["inputs"]
        )
        ex["inputs"] = torch.from_numpy(
            np.array(spectrograms.compute_spectrogram(samples, self.spectrogram_config))
        )
        # add normalization
        ex["inputs"] = torch.clamp(ex["inputs"], min=MIN_LOG_MEL, max=MAX_LOG_MEL)
        ex["inputs"] = (ex["inputs"] - MIN_LOG_MEL) / (MAX_LOG_MEL - MIN_LOG_MEL)
        return ex

    def _pad_length(self, row):
        inputs = row["inputs"][: self.mel_length].to(torch.float32)
        targets = torch.from_numpy(row["targets"][: self.event_length]).to(torch.long)
        targets = targets + self.vocab.num_special_tokens()
        if inputs.shape[0] < self.mel_length:
            pad = torch.zeros(
                self.mel_length - inputs.shape[0],
                inputs.shape[1],
                dtype=inputs.dtype,
                device=inputs.device,
            )
            inputs = torch.cat([inputs, pad], dim=0)
        if targets.shape[0] < self.event_length:
            eos = torch.ones(1, dtype=targets.dtype, device=targets.device)
            if self.event_length - targets.shape[0] - 1 > 0:
                pad = (
                    torch.ones(
                        self.event_length - targets.shape[0] - 1,
                        dtype=targets.dtype,
                        device=targets.device,
                    )
                    * -100
                )
                targets = torch.cat([targets, eos, pad], dim=0)
            else:
                targets = torch.cat([targets, eos], dim=0)
        return {"inputs": inputs, "targets": targets, "input_times": row["input_times"]}

    def _split_frame(self, row, length=2000):
        rows = []
        input_length = row["inputs"].shape[0]

        # chunk the audio into chunks of `length` = 2000
        # during _random_chunk, within each chunk, randomly select a segment = self.mel_length
        for split in range(0, input_length, length):
            if split + length >= input_length:
                continue
            new_row = {}
            for k in row.keys():
                if k in [
                    "inputs",
                    "input_times",
                    "input_event_start_indices",
                    "input_event_end_indices",
                    "input_state_event_indices",
                ]:
                    new_row[k] = row[k][split : split + length]
                else:
                    new_row[k] = row[k]
            rows.append(new_row)

        if len(rows) == 0:
            return [row]
        return rows

    def _random_chunk(self, row):
        new_row = {}
        input_length = row["inputs"].shape[0]
        random_length = input_length - self.mel_length
        if random_length < 1:
            return row
        if self.is_deterministic:
            start_length = 0
        else:
            start_length = random.randint(0, random_length)
        for k in row.keys():
            if k in [
                "inputs",
                "input_times",
                "input_event_start_indices",
                "input_event_end_indices",
                "input_state_event_indices",
            ]:
                new_row[k] = row[k][start_length : start_length + self.mel_length]
            else:
                new_row[k] = row[k]
        return new_row

    def _to_event(self, predictions_np: List[np.ndarray], frame_times: np.ndarray):
        predictions = []
        for i, batch in enumerate(predictions_np):
            for j, tokens in enumerate(batch):
                # tokens = tokens[:np.argmax(
                #     tokens == vocabularies.DECODED_EOS_ID)]
                start_time = frame_times[i][j][0]
                start_time -= start_time % (1 / self.codec.steps_per_second)
                predictions.append(
                    {"est_tokens": tokens, "start_time": start_time, "raw_inputs": []}
                )

        encoding_spec = note_sequences.NoteEncodingWithTiesSpec
        result = metrics_utils.event_predictions_to_ns(
            predictions, codec=self.codec, encoding_spec=encoding_spec
        )
        return result["est_ns"]

    def _postprocess_batch(self, result):
        EOS_TOKEN_ID = 1  # TODO: this is a hack!
        after_eos = torch.cumsum((result == EOS_TOKEN_ID).float(), dim=-1)
        # minus special token
        result = result - self.vocab.num_special_tokens()
        result = torch.where(after_eos.bool(), -1, result)
        # remove bos
        result = result[:, 1:]
        result = result.cpu().numpy()
        return result

    def _preprocess_inputs(self, row):
        ns = self._parse_midi(row["midi_path"])
        audio, sr = librosa.load(row["audio_path"], sr=None)
        if sr != self.spectrogram_config.sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=self.spectrogram_config.sample_rate
            )

        return ns, audio
    
    def __getitem__(self, idx):
        ns, audio = self._preprocess_inputs(self.df[idx])

        row = self._tokenize(ns, audio)

        # NOTE: if we need to ensure that the chunks in `rows` to be contiguous,
        # use `length = self.mel_length` in `_split_frame`:
        rows = self._split_frame(row, length=self.split_frame_length)

        inputs, targets, frame_times = [], [], []
        if len(rows) > self.num_rows_per_batch:
            if self.is_deterministic:
                start_idx = 0
            else:
                start_idx = random.randint(0, len(rows) - self.num_rows_per_batch)
            rows = rows[start_idx : start_idx + self.num_rows_per_batch]

        for j, row in enumerate(rows):
            row = self._random_chunk(row)
            row = self._extract_target_sequence_with_indices(row, self.tie_token)
            row = self._run_length_encode_shifts(row)

            row = self._compute_spectrogram(row)

            # -- random order augmentation --
            if self.is_randomize_tokens:
                t = self.randomize_tokens(
                    [self.get_token_name(t) for t in row["targets"]]
                )
                t = np.array([self.token_to_idx(k) for k in t])
                t = self._remove_redundant_tokens(t)
                row["targets"] = t

            row = self._pad_length(row)
            inputs.append(row["inputs"])
            targets.append(row["targets"])

        return torch.stack(inputs), torch.stack(targets)

    def __len__(self):
        return len(self.df)

    def randomize_tokens(self, token_lst):
        shift_idx = [i for i in range(len(token_lst)) if "shift" in token_lst[i]]
        if len(shift_idx) == 0:
            return token_lst
        res = token_lst[: shift_idx[0]]
        for j in range(len(shift_idx) - 1):
            res += [token_lst[shift_idx[j]]]

            start_idx = shift_idx[j]
            end_idx = shift_idx[j + 1]
            cur = token_lst[start_idx + 1 : end_idx]
            cur_lst = []
            ptr = 0
            while ptr < len(cur):
                t = cur[ptr]
                if "program" in t:
                    cur_lst.append([cur[ptr], cur[ptr + 1], cur[ptr + 2]])
                    ptr += 3
                elif "velocity" in t:
                    cur_lst.append([cur[ptr], cur[ptr + 1]])
                    ptr += 2

            indices = np.arange(len(cur_lst))
            np.random.shuffle(indices)

            new_cur_lst = []
            for idx in indices:
                new_cur_lst.append(cur_lst[idx])

            new_cur_lst = [item for sublist in new_cur_lst for item in sublist]
            res += new_cur_lst

        res += token_lst[shift_idx[-1] :]
        return res

    def get_token_name(self, token_idx):
        token_idx = int(token_idx)
        if token_idx >= 1001 and token_idx <= 1128:
            token = f"pitch_{token_idx - 1001}"
        elif token_idx >= 1129 and token_idx <= 1130:
            token = f"velocity_{token_idx - 1129}"
        elif token_idx >= 1131 and token_idx <= 1131:
            token = "tie"
        elif token_idx >= 1132 and token_idx <= 1259:
            token = f"program_{token_idx - 1132}"
        elif token_idx >= 1260 and token_idx <= 1387:
            token = f"drum_{token_idx - 1260}"
        elif token_idx >= 0 and token_idx < 1000:
            token = f"shift_{token_idx}"
        else:
            token = f"invalid_{token_idx}"

        return token

    def token_to_idx(self, token_str):
        if "pitch" in token_str:
            token_idx = int(token_str.split("_")[1]) + 1001
        elif "velocity" in token_str:
            token_idx = int(token_str.split("_")[1]) + 1129
        elif "tie" in token_str:
            token_idx = 1131
        elif "program" in token_str:
            token_idx = int(token_str.split("_")[1]) + 1132
        elif "drum" in token_str:
            token_idx = int(token_str.split("_")[1]) + 1260
        elif "shift" in token_str:
            token_idx = int(token_str.split("_")[1])

        return token_idx


def collate_fn(lst):
    inputs = torch.cat([k[0] for k in lst])
    targets = torch.cat([k[1] for k in lst])
    return inputs, targets



if __name__ == "__main__":
    dataset = Dataset(
        root_dir="/home/chou150/depot/datasets/maestro/maestro_with_mistakes",
        split_json_path="/home/chou150/depot/datasets/maestro/maestro-v3.0.0/maestro-v3.0.0.json",
        split="train",
        shuffle=False,
        is_train=False,
        include_ties=True,
        mel_length=256,
        split_frame_length=512,  # was 256 for some reason
        num_rows_per_batch=127,
        is_deterministic=True,  # True
        is_randomize_tokens=True,
    )
    print("pitch", dataset.codec.event_type_range("pitch"))
    print("velocity", dataset.codec.event_type_range("velocity"))
    print("tie", dataset.codec.event_type_range("tie"))
    print("error_class", dataset.codec.event_type_range("error_class"))
    dl = DataLoader(
        dataset, batch_size=1, num_workers=1, collate_fn=collate_fn, shuffle=False
    )
    EOS_TOKEN_ID = 1
    # Define the EOS token outside the loop for efficiency.

    for idx, item in enumerate(dl):

        mistake_inputs, score_inputs, targets = item

        print("targets", targets.shape, flush=True)
        print("mistake_inputs", mistake_inputs.shape, flush=True)
        print("score_inputs", score_inputs.shape, flush=True)
        print("targets", targets, flush=True)

        # Convert tensor to numpy array if necessary (especially if using GPU).
        targets_numpy = (
            targets.cpu().numpy() if torch.cuda.is_available() else targets.numpy()
        )

        # Filter out the EOS token and decode other tokens.
        for token in targets_numpy[0]:
            token_name = dataset.get_token_name(token - 3)
            print("token", token_name, flush=True)

            if token == EOS_TOKEN_ID:
                break

        # Process and plot mistake_inputs
        mel_spectrogram_mistake = (
            mistake_inputs.squeeze().cpu().numpy()
            if torch.cuda.is_available()
            else mistake_inputs.squeeze().numpy()
        )
        if mel_spectrogram_mistake.ndim == 3:
            mel_spectrogram_mistake = mel_spectrogram_mistake[
                1
            ]  # select the first spectrogram from batch
        if mel_spectrogram_mistake.ndim == 2:
            plot_spectrogram(
                mel_spectrogram_mistake,
                idx,
                "Mel Spectrogram (Mistake Inputs)",
                "mel_spectrogram_mistake",
            )

        # Process and plot score_inputs
        mel_spectrogram_score = (
            score_inputs.squeeze().cpu().numpy()
            if torch.cuda.is_available()
            else score_inputs.squeeze().numpy()
        )
        if mel_spectrogram_score.ndim == 3:
            mel_spectrogram_score = mel_spectrogram_score[
                1
            ]  # select the first spectrogram from batch
        if mel_spectrogram_score.ndim == 2:
            plot_spectrogram(
                mel_spectrogram_score,
                idx,
                "Mel Spectrogram (Score Inputs)",
                "mel_spectrogram_score",
            )

        if idx == 0:  # Stop after processing the first batch item
            break
        # decoded_tokens = [
        #     dataset.get_token_name(token)
        #     for token in targets_numpy[0]  # Assuming single batch handling
        #     if token != EOS_TOKEN_ID
        # ]
        # print("mistake_inputs", mistake_inputs, flush=True)
        # print("score_inputs", score_inputs, flush=True)
        # np.save("mt3_0001_label.npy", targets.cpu().numpy())
        # print(idx, inputs.shape, targets[0][:100])
        break
