# Modified from the original work: Copyright 2022 The MT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This code has been modified from the original MT3 repository for Polytune.
# The original repository can be found at: https://github.com/[original-author]/mt3
#
# This software is provided on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""Transcription preprocessors."""

from typing import Tuple

from immutabledict import immutabledict
import note_seq

ERROR_CLASSES = {
    "Correct": 0,
    "Missing": 1,
    "Extra": 2,
}


def class_to_error(slakh_class: str) -> Tuple[int, bool]:
    """Map Slakh class to an error class."""

    if slakh_class in ERROR_CLASSES:
        return error_classes[slakh_class]
    else:
        raise ValueError(f"unknown Slakh class: {slakh_class}")


class PitchBendError(Exception):
    pass


def add_track_to_notesequence(
    ns: note_seq.NoteSequence,
    track: note_seq.NoteSequence,
    error_class: int,  # Changed 'program' to 'error_class'
    ignore_pitch_bends: bool,
):
    """Add a track to a NoteSequence with error class instead of program."""
    if track.pitch_bends and not ignore_pitch_bends:
        raise PitchBendError("Pitch bends not supported")
    track_sus = note_seq.apply_sustain_control_changes(track)
    for note in track_sus.notes:
        note.instrument = error_class  # Using 'instrument' to store error class
        ns.notes.extend([note])
        ns.total_time = max(ns.total_time, note.end_time)
