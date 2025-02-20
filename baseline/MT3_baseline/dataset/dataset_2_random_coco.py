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
import re
import matplotlib.pyplot as plt
import numpy as np
import json
import random
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import librosa
import note_seq
from glob import glob
from contrib import (
    event_codec,
    note_sequences,
    spectrograms,
    vocabularies,
    run_length_encoding,
    metrics_utils,
)
from contrib.preprocessor import (
    class_to_error,
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
        is_random_alignment_shift_augmentation=False,
        is_deterministic=False,
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
        output_json_path = os.path.join(os.path.dirname(split_json_path), "file_paths_8_12.json")
        self.df = self._build_dataset(root_dir, split_json_path, split, output_json_path, shuffle=shuffle)
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
        self.random_alignment_shift_augmentation = is_random_alignment_shift_augmentation
        self.context_multiplier = 2

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


    def _build_dataset(self, root_dir, json_path, split, output_json_file, shuffle=True):
        # Load the mapping and splits
        midi_filename_to_number, split_to_numbers = self._load_MAESTRO_split_info(json_path)
        print(f"Finished loading {json_path}", flush=True)
        desired_file_numbers = split_to_numbers[split]

        df = []
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
                        if key in ["extra_notes", "removed_notes", "correct_notes"]:
                            if filename.endswith(".mid"):
                                file_path = os.path.join(root, filename)
                                dir_key = os.path.normpath(file_path).split(os.sep)[-3]
                                if dir_key not in files_dict[key]:
                                    files_dict[key][dir_key] = []
                                files_dict[key][dir_key].append(file_path)
                                batch.append(file_path)
                                print(f"file_path: {file_path}", flush=True)
                        else:
                            if not filename.endswith("mix.mid") and not filename.endswith("mix.wav"):
                                if filename.endswith(".mid") or filename.endswith(".wav"):
                                    file_path = os.path.join(root, filename)
                                    dir_key = os.path.normpath(file_path).split(os.sep)[-3]
                                    print(f"dir_key: {dir_key}", flush=True)
                                    if dir_key not in files_dict[key]:
                                        files_dict[key][dir_key] = []
                                    files_dict[key][dir_key].append(file_path)
                                    batch.append(file_path)
                                    print(f"file_path: {file_path}", flush=True)

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
            files_dict = load_paths_from_json(output_json_file)
        else:
            files_dict = scan_and_save_paths(directories, output_json_file)

        # Example usage: accessing file paths from the loaded dictionary
        extra_notes_files = files_dict["extra_notes"]
        removed_notes_files = files_dict["removed_notes"]
        correct_notes_files = files_dict["correct_notes"]
        mistake_files = files_dict["mistake"]
        score_files = files_dict["score"]

        
        # Match files based on the common identifier
        for track_id in extra_notes_files.keys():
            file_number = midi_filename_to_number.get(track_id)
            if file_number in desired_file_numbers:
                # Ensure all file types have the same number of subtracks
                num_subtracks = len(extra_notes_files[track_id])
                if track_id in removed_notes_files and track_id in correct_notes_files and track_id in mistake_files and track_id in score_files:
                    # Iterate over each subtrack index
                    for i in range(num_subtracks):
                        # Ensure that all data types have this subtrack index
                        if (i < len(removed_notes_files[track_id]) and i < len(correct_notes_files[track_id]) and
                            i < len(mistake_files[track_id]) and i < len(score_files[track_id])):
                            df.append({
                                "extra_notes_midi": extra_notes_files[track_id][i],
                                "removed_notes_midi": removed_notes_files[track_id][i],
                                "correct_notes_midi": correct_notes_files[track_id][i],
                                "mistake_audio": self.capitalize_instrument_name(mistake_files[track_id][i].replace("stems_midi", "stems_audio").replace(".mid", ".wav")),
                                "score_audio": self.capitalize_instrument_name(score_files[track_id][i].replace("stems_midi", "stems_audio").replace(".mid", ".wav"))
                            })

        assert len(df) > 0, "No matching files found. Check the dataset directory."
        print("Total files:", len(df), flush=True)
        
        # Shuffle the dataset if needed
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

    # def _parse_midi(self, path, instrument_dict: Dict[str, str]):
    #     """
    #     __getitem__ loads the MIDIfiles as a list of note sequences and a list of instrument values.
    #     Parses MIDI files and returns note sequences and instrument values.

    #     Args:
    #         path (str): The path to the directory containing the MIDI files.
    #         instrument_dict (Dict[str, str]): A dictionary mapping filenames to instrument values.

    #     Returns:
    #         Tuple[List[NoteSequence], List[str]]: A tuple containing a list of note sequences and a list of instrument values.
    #     """
    #     note_seqs = []

    #     for filename in instrument_dict.keys():
    #         # this can be pretty_midi.PrettyMIDI() obj / string path to midi
    #         midi_path = f"{path}/{filename}.mid"
    #         note_seqs.append(note_seq.midi_file_to_note_sequence(midi_path))
    #     return note_seqs, instrument_dict.values()

    def _tokenize(
        self,
        tracks: List[note_seq.NoteSequence],
        mistake_samples: np.ndarray,
        score_samples: np.ndarray,
        example_id: Optional[str] = None,  # not used
    ):

        frames_mistake, frame_times_mistake = self._audio_to_frames(mistake_samples)
        frames_score, frame_times_score = self._audio_to_frames(score_samples)

        # Add all the notes from the tracks to a single NoteSequence.
        ns = note_seq.NoteSequence(ticks_per_quarter=220)
        # tracks = [note_seq.NoteSequence.FromString(seq) for seq in sequences]
        # print("tracks", len(tracks))
        for i, track in enumerate(tracks):
            # TODO: mapping of error_classes
            error_class = (
                i + 1
            )  # TODO: NOTE: This might be wrong, need to adjust!!!!!!!!
            try:
                add_track_to_notesequence(
                    ns,
                    track,
                    error_class=error_class,
                    ignore_pitch_bends=self.ignore_pitch_bends,
                )
            except PitchBendError:
                # TODO(iansimon): is there a way to count these?
                return

        note_sequences.assign_error_classes(ns)
        note_sequences.validate_note_sequence(ns)
        if self.is_train:
            # Trim overlapping notes in training (as our event vocabulary cannot
            # represent them), but preserve original NoteSequence for eval.
            ns = note_sequences.trim_overlapping_notes(ns)

        if example_id is not None:
            ns.id = example_id

        if self.onsets_only:
            times, values = note_sequences.note_sequence_to_onsets(
                ns
            )  # get the start time of each note
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
            frame_times=frame_times_mistake,
            encoding_state_to_events_fn=(
                note_sequences.note_encoding_state_to_events
                if self.include_ties
                else None
            ),
        )

        # print("events", events)
        # print('inputs', np.array(frames).shape)
        # print('frame_times', frame_times.shape)
        # print("targets", events.shape)
        return {
            "mistake_inputs": np.array(frames_mistake),
            "mistake_input_times": frame_times_mistake,
            "score_inputs": np.array(frames_score),
            "score_input_times": frame_times_score,
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
        # print(
        #     "features[targets]", [self.get_token_name(k) for k in features["targets"]]
        # )

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
                # Ensure we don't run off the end of the state events. # TODO: this is a hack!
                if state_event_end_idx >= len(features["state_events"]):
                    print("encountered end of state events without end token", flush=True)
                    break
                
            features["targets"] = np.concatenate(
                [
                    features["state_events"][state_event_start_idx:state_event_end_idx],
                    features["targets"],
                ],
                axis=0,
            )
            # print(
            #     "features[targets] 2",
            #     [self.get_token_name(k) for k in features["targets"]],
            # )

        return features

    def _run_length_encode_shifts(
        self,
        features,
        state_change_event_types=["velocity", "error_class"],
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
        return features

    def _remove_redundant_tokens(
        self,
        events,
        state_change_event_types=["velocity", "error_class"],
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
        # Process 'mistake_inputs'
        mistake_samples = spectrograms.flatten_frames(ex["mistake_inputs"])
        if mistake_samples.shape[0] == 0:
            # assign a default empty waveform of 0's to the mistake samples
            # this is a temporary fix to handle the case where the mistake samples are empty
            mistake_samples = np.zeros(
                self.mel_length * self.spectrogram_config.hop_width
            )

        # Check if the length is less than required for the FFT
        if mistake_samples.shape[0] < self.spectrogram_config.n_fft:
            pad_width = self.spectrogram_config.n_fft - mistake_samples.shape[0]
            mistake_samples = np.pad(mistake_samples, (0, pad_width), mode='constant')


        # Compute spectrogram
        ex["mistake_inputs"] = torch.from_numpy(
            np.array(
                spectrograms.compute_spectrogram(
                    mistake_samples, self.spectrogram_config
                )
            )
        )

        # Normalize the spectrogram
        ex["mistake_inputs"] = torch.clamp(
            ex["mistake_inputs"], min=MIN_LOG_MEL, max=MAX_LOG_MEL
        )
        ex["mistake_inputs"] = (ex["mistake_inputs"] - MIN_LOG_MEL) / (
            MAX_LOG_MEL - MIN_LOG_MEL
        )

        # Process 'score_inputs' similarly
        score_samples = spectrograms.flatten_frames(ex["score_inputs"])
        if score_samples.shape[0] == 0:
            # assign a default empty waveform of 0's to the score samples
            # this is a temporary fix to handle the case where the score samples are empty
            score_samples = np.zeros(
                self.mel_length * self.spectrogram_config.hop_width
            )


        if score_samples.shape[0] < self.spectrogram_config.n_fft:
            pad_width = self.spectrogram_config.n_fft - score_samples.shape[0]
            score_samples = np.pad(score_samples, (0, pad_width), mode='constant')


        # Compute spectrogram
        ex["score_inputs"] = torch.from_numpy(
            np.array(
                spectrograms.compute_spectrogram(
                    score_samples,
                    self.spectrogram_config,
                    context_multiplier=self.context_multiplier,
                )
            )
        )

        # Normalize the spectrogram
        ex["score_inputs"] = torch.clamp(
            ex["score_inputs"], min=MIN_LOG_MEL, max=MAX_LOG_MEL
        )
        ex["score_inputs"] = (ex["score_inputs"] - MIN_LOG_MEL) / (
            MAX_LOG_MEL - MIN_LOG_MEL
        )

        return ex


    def _pad_length(self, row):
        # Pad mistake inputs
        mistake_inputs = row["mistake_inputs"][: self.mel_length].to(torch.float32)
        if mistake_inputs.shape[0] < self.mel_length:
            pad = torch.zeros(
                self.mel_length - mistake_inputs.shape[0],
                mistake_inputs.shape[1],
                dtype=mistake_inputs.dtype,
                device=mistake_inputs.device,
            )
            mistake_inputs = torch.cat([mistake_inputs, pad], dim=0)

        # Pad score inputs
        score_inputs = row["score_inputs"][: self.mel_length].to(torch.float32)
        if score_inputs.shape[0] < self.mel_length:
            pad = torch.zeros(
                self.mel_length - score_inputs.shape[0],
                score_inputs.shape[1],
                dtype=score_inputs.dtype,
                device=score_inputs.device,
            )
            score_inputs = torch.cat([score_inputs, pad], dim=0)

        # Pad targets
        targets = torch.from_numpy(row["targets"][: self.event_length]).to(torch.long)
        targets = targets + self.vocab.num_special_tokens()
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
                
        if row["mistake_input_times"].ndim == 0:
            print("dim error: mistake_input_times", row["mistake_input_times"], flush=True)
            row["mistake_input_times"] = row["mistake_input_times"].unsqueeze(0)
            
        if row["score_input_times"].ndim == 0:
            print("dim error: score_input_times", row["score_input_times"], flush=True)
            row["score_input_times"] = row["score_input_times"].unsqueeze(0)
            
        # Return padded data
        return {
            "mistake_inputs": mistake_inputs,
            "score_inputs": score_inputs,
            "targets": targets,
            "mistake_input_times": row["mistake_input_times"][: self.mel_length],
            "score_input_times": row["score_input_times"][: self.mel_length],
        }

    def _split_frame(self, row, length=2000):
        rows = []
        input_length = row["score_inputs"].shape[0]

        # chunk the audio into chunks of `length` = 2000
        # during _random_chunk, within each chunk, randomly select a segment = self.mel_length
        for split in range(0, input_length, length):
            if split + length >= input_length:
                continue
            new_row = {}
            for k in row.keys():
                if k in [
                    "mistake_inputs",
                    "mistake_input_times",
                    "score_inputs",
                    "score_input_times",
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

    # check padding for non score inputs
    def _random_chunk(self, row):
        input_length = row["mistake_inputs"].shape[0]
        # print("input_length", input_length)
        random_length = input_length - self.mel_length

        if random_length < 1:
            start_length = 0  # Start from the beginning if not enough length
            padding_length = self.mel_length - input_length
        else:
            start_length = (
                random.randint(0, random_length) if not self.is_deterministic else 0
            )
            padding_length = 0

        # Calculate the total amount of context needed around the slice
        extra_context_window = self.context_multiplier - 1
        extended_length = self.mel_length * self.context_multiplier
        if self.random_alignment_shift_augmentation:
            random_shift = random.randint(
                -int(self.mel_length * 0.5), int(self.mel_length * 0.5)
            )
        else:
            random_shift = 0
        # Calculate start and end indices for extended context
        extended_start = max(
            0, start_length - int(self.mel_length * 0.5 * extra_context_window) + random_shift
        )
        extended_end = min(
            input_length,
            start_length
            + self.mel_length
            + int(self.mel_length * 0.5 * extra_context_window) + random_shift,
        )
        # print("extended_start", extended_start)
        # print("extended_end", extended_end)
        # for no extended context uncomment the following
        # extended_start = start_length
        # extended_end = start_length + self.mel_length

        # Calculate padding if necessary
        start_padding = extended_start - (
            start_length - int(self.mel_length * 0.5 * extra_context_window)
        )
        end_padding = (
            start_length
            + self.mel_length
            + int(self.mel_length * 0.5 * extra_context_window)
        ) - extended_end
        # for no extended context uncomment the following
        # start_padding = int(self.mel_length * extra_context_window * 0.5)
        # end_padding = int(self.mel_length * extra_context_window * 0.5)

        new_row = {}
        for k in row.keys():
            if k in [
                "mistake_inputs",
                "mistake_input_times",
                "input_event_start_indices",
                "input_event_end_indices",
                "input_state_event_indices",
            ]:
                # Standard context without extended padding
                slice_start = start_length
                slice_end = start_length + self.mel_length
                data_slice = row[k][slice_start:slice_end]
                if isinstance(data_slice, np.ndarray):
                    data_slice = torch.from_numpy(data_slice)
                if padding_length > 0:
                    if len(data_slice.shape) == 2:
                        padding = (0, 0, 0, padding_length)
                    if len(data_slice.shape) == 1:
                        padding = (0, padding_length)
                    data_slice = F.pad(data_slice, padding, "constant", 0).squeeze(0)
            elif k in [
                "score_inputs",
                "score_input_times",
            ]:  # Apply extended context for score inputs
                data_slice = row[k][extended_start:extended_end]
                # print(f"data slice before padding for {k}", data_slice.shape)
                if isinstance(data_slice, np.ndarray):
                    data_slice = torch.from_numpy(data_slice)
                # Apply padding if necessary
                if start_padding > 0 or end_padding > 0:
                    # print("padding", start_padding, end_padding)
                    if len(data_slice.shape) == 2:
                        padding = (0, 0, max(0, start_padding), max(0, end_padding))
                    else:
                        padding = (max(0, start_padding), max(0, end_padding))
                    data_slice = F.pad(data_slice, padding, "constant", 0).squeeze(0)

                # print(f"data slice after padding for {k}", data_slice.shape)
            else:
                data_slice = row[k]

            new_row[k] = data_slice

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
        """
        Load and parse MIDI and audio files for given dataset entry.

        Args:
            row (dict): Dictionary containing paths to MIDI and audio files.

        Returns:
            Tuple containing:
                - A list of note sequences parsed from the MIDI files.
                - The audio waveform corresponding to the 'mistake' audio file.
                - The audio waveform corresponding to the 'score' audio file.
        """
        # Example check for 'mistake_audio'
        mistake_audio_path = row.get("mistake_audio")
        if mistake_audio_path and os.path.exists(mistake_audio_path):
            mistake_audio, sr_mistake = librosa.load(mistake_audio_path, sr=None)
        else:
            print(f"File not found: {mistake_audio_path}")
            return [None, None, None], None, None

        # Repeat the check for 'score_audio'
        score_audio_path = row.get("score_audio")
        if score_audio_path and os.path.exists(score_audio_path):
            score_audio, sr_score = librosa.load(score_audio_path, sr=None)
        else:
            print(f"File not found: {score_audio_path}")
            return [None, None, None], None, None
        extra_notes_ns = note_seq.midi_file_to_note_sequence(row["extra_notes_midi"])
        removed_notes_ns = note_seq.midi_file_to_note_sequence(
            row["removed_notes_midi"]
        )
        mistake_ns = note_seq.midi_file_to_note_sequence(row["correct_notes_midi"])

        # Load audio files
        mistake_audio, sr_mistake = librosa.load(row["mistake_audio"], sr=None)
        score_audio, sr_score = librosa.load(row["score_audio"], sr=None)

        # Resample audio if necessary
        if sr_mistake != self.spectrogram_config.sample_rate:
            mistake_audio = librosa.resample(
                mistake_audio,
                orig_sr=sr_mistake,
                target_sr=self.spectrogram_config.sample_rate,
            )
        if sr_score != self.spectrogram_config.sample_rate:
            score_audio = librosa.resample(
                score_audio,
                orig_sr=sr_score,
                target_sr=self.spectrogram_config.sample_rate,
            )

        # Return tuple of note sequences and audio data
        return (
            [extra_notes_ns, removed_notes_ns, mistake_ns],
            mistake_audio,
            score_audio,
        )

    def __getitem__(self, idx):

        # print("getting item", idx, flush=True)
        (extra_notes_ns, removed_notes_ns, mistake_ns), mistake_audio, score_audio = (
            self._preprocess_inputs(self.df[idx])
        )
        if mistake_audio is None or score_audio is None:
            return None
        # print("preprocessed inputs", flush=True)
        row = self._tokenize(
            [extra_notes_ns, removed_notes_ns, mistake_ns],
            mistake_audio,
            score_audio,
            None,
        )
        # print("tokenized", flush=True)
        # NOTE: if we need to ensure that the chunks in `rows` to be contiguous,
        # use `length = self.mel_length` in `_split_frame`:
        rows = self._split_frame(row, length=self.split_frame_length)
        # print("split frames", flush=True)
        (
            mistake_inputs,
            targets,
            mistake_frame_times,
            score_inputs,
            score_frame_times,
        ) = ([], [], [], [], [])
        if len(rows) > self.num_rows_per_batch:
            if self.is_deterministic:
                start_idx = 0
            else:
                start_idx = random.randint(0, len(rows) - self.num_rows_per_batch)
            rows = rows[start_idx : start_idx + self.num_rows_per_batch]

        for j, row in enumerate(rows):
            row = self._random_chunk(row)
            # print("random chunk", flush=True)
            row = self._extract_target_sequence_with_indices(row, self.tie_token)
            # print("extracted target sequence", flush=True)
            row = self._run_length_encode_shifts(row)
            # print("run length encoded shifts", flush=True)
            row = self._compute_spectrogram(row)
            # print("computed spectrogram", flush=True)

            # -- random order augmentation --
            if self.is_randomize_tokens:
                t = self.randomize_tokens(
                    [self.get_token_name(t) for t in row["targets"]]
                )
                t = np.array([self.token_to_idx(k) for k in t])
                t = self._remove_redundant_tokens(t)
                row["targets"] = t
            
            
                
            # print("randomized tokens", flush=True)
            row = self._pad_length(row)
            # print("padded length", flush=True)
            mistake_inputs.append(row["mistake_inputs"])
            score_inputs.append(row["score_inputs"])
            targets.append(row["targets"])

        return (
            torch.stack(mistake_inputs),
            torch.stack(score_inputs),
            torch.stack(targets),
        )

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
            # print("cur", cur)
            # print(f"len(cur) {len(cur)}")
            while ptr < len(cur):
                t = cur[ptr]
                if "error" in t:
                    # print("error_class", t)
                    cur_lst.append([cur[ptr], cur[ptr + 1], cur[ptr + 2]])
                    ptr += 3
                elif "velocity" in t:
                    # print("velocity", t)
                    cur_lst.append([cur[ptr], cur[ptr + 1]])
                    ptr += 2
                else:
                    raise Exception("infinte loop detected, check get_token_name names")
                # print("ptr", ptr)

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
            return f"pitch_{token_idx - 1001}"
        elif token_idx >= 1129 and token_idx <= 1130:
            return f"velocity_{token_idx - 1129}"
        elif token_idx == 1131:
            return "tie"
        elif token_idx >= 1132 and token_idx <= 1134:
            return f"error_{token_idx - 1132}"  # Adjusted to the new range for 3 error types
        elif token_idx >= 0 and token_idx < 1000:
            return f"shift_{token_idx}"
        else:
            return f"invalid_{token_idx}"

        return token

    def token_to_idx(self, token_str):
        if "pitch" in token_str:
            token_idx = int(token_str.split("_")[1]) + 1001
        elif "velocity" in token_str:
            token_idx = int(token_str.split("_")[1]) + 1129
        elif "tie" in token_str:
            token_idx = 1131
        elif "error" in token_str:
            token_idx = int(token_str.split("_")[1]) + 1132
        elif "shift" in token_str:
            token_idx = int(token_str.split("_")[1])
        else:
            raise ValueError("Unknown token string: {}".format(token_str))

        return token_idx


def collate_fn(lst):
    # Filter out None entries which may be returned if files are missing or any error occurs
    filtered_lst = [item for item in lst if item is not None]

    if not filtered_lst:
        # Return None or raise an exception if all entries are None, or provide dummy data
        # Returning None might require your training loop to handle this case
        return None

    # If there are valid items, concatenate them
    mistake_inputs = torch.cat([item[0] for item in filtered_lst], dim=0)
    score_inputs = torch.cat([item[1] for item in filtered_lst], dim=0)
    targets = torch.cat([item[2] for item in filtered_lst], dim=0)

    return mistake_inputs, score_inputs, targets



def plot_spectrogram(spectrogram, idx, title, file_suffix):
    plt.figure(figsize=(10, 4))
    # Transpose the spectrogram to have time on the x-axis and frequency on the y-axis
    spectrogram = spectrogram.T  # Transpose to flip axes
    plt.imshow(spectrogram, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"{title} for Sample {idx}")
    plt.xlabel("Time Frame")
    plt.ylabel("Mel Frequency")
    plt.savefig(f"{file_suffix}_{idx}.png")
    plt.close()
    print(f"Saved {title.lower()} image for sample {idx}.")


if __name__ == "__main__":
    dataset = Dataset(
        root_dir="/home/chou150/depot/datasets/maestro/maestro_with_mistakes",
        split_json_path="/home/chou150/depot/datasets/maestro/maestro-v3.0.0/maestro-v3.0.0.json",
        split="train",
        shuffle=False,
        is_train=False,
        include_ties=True,
        mel_length=256,
        split_frame_length=30000,  # was 256 for some reason
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
