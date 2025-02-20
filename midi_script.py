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
import miditoolkit
import glob
import os
from tqdm import tqdm

"""
Use this for evaluation!!!

This script processes MIDI files located in the specified directory and creates a new MIDI file that contains all the instruments from the input MIDI files. The new MIDI file is saved with the name "all_src_v2.mid" in the same directory as the input MIDI files.

The script performs the following steps:
1. Retrieves a list of MIDI files located in the specified directory.
2. For each MIDI file, extracts the individual instrument tracks.
3. Creates a new MIDI object and sets its properties (ticks per beat, time signature changes, tempo changes, and key signature changes) based on the first MIDI file.
4. Adds all the instrument tracks to the new MIDI object.
5. Saves the new MIDI object as "all_src_v2.mid" in the same directory as the input MIDI files.

"""
for item in ["train", "validation", "test"]:
    midis = sorted(
        glob.glob(
            f"/depot/yunglu/data/datasets_ben/MR_MT3/slakh2100_flac_redux/{item}/*/MIDI/"
        )
    )
    for midi in tqdm(midis):
        stems = sorted(glob.glob(midi + "*.mid"))
        insts = []
        for stem in stems:
            midi_obj = miditoolkit.MidiFile(stem)
            for inst in midi_obj.instruments:
                insts.append(inst)

        new_midi_obj = miditoolkit.MidiFile()
        new_midi_obj.ticks_per_beat = midi_obj.ticks_per_beat
        new_midi_obj.time_signature_changes = midi_obj.time_signature_changes
        new_midi_obj.tempo_changes = midi_obj.tempo_changes
        new_midi_obj.key_signature_changes = midi_obj.key_signature_changes
        new_midi_obj.instruments = insts

        new_midi_obj.dump(os.path.join(midi.replace("MIDI/", ""), "all_src_v2.mid"))
# midis = sorted(
#     glob.glob(
#         "/depot/yunglu/data/datasets_ben/MR_MT3/slakh2100_flac_redux/test/*/MIDI/"
#     )
# )
# for midi in tqdm(midis):
#     stems = sorted(glob.glob(midi + "*.mid"))
#     insts = []
#     for stem in stems:
#         midi_obj = miditoolkit.MidiFile(stem)
#         for inst in midi_obj.instruments:
#             insts.append(inst)

#     new_midi_obj = miditoolkit.MidiFile()
#     new_midi_obj.ticks_per_beat = midi_obj.ticks_per_beat
#     new_midi_obj.time_signature_changes = midi_obj.time_signature_changes
#     new_midi_obj.tempo_changes = midi_obj.tempo_changes
#     new_midi_obj.key_signature_changes = midi_obj.key_signature_changes
#     new_midi_obj.instruments = insts

#     new_midi_obj.dump(os.path.join(midi.replace("MIDI/", ""), "all_src_v2.mid"))
