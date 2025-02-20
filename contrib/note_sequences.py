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

"""Helper functions that operate on NoteSequence protos."""

import dataclasses
import itertools

from typing import MutableMapping, MutableSet, Optional, Sequence, Tuple
from contrib import event_codec, vocabularies, run_length_encoding
import note_seq

DEFAULT_VELOCITY = 100
DEFAULT_NOTE_DURATION = 0.01

# Quantization can result in zero-length notes; enforce a minimum duration.
MIN_NOTE_DURATION = 0.01


@dataclasses.dataclass
class TrackSpec:
    name: str
    error_class: int = 0  # Now represents error classes


def extract_track(ns, error_class):
    track = note_seq.NoteSequence(ticks_per_quarter=220)
    track_notes = [note for note in ns.notes if note.instrument == error_class]
    track.notes.extend(track_notes)
    track.total_time = (
        max(note.end_time for note in track.notes) if track.notes else 0.0
    )
    return track


def trim_overlapping_notes(ns: note_seq.NoteSequence) -> note_seq.NoteSequence:
    """Trim overlapping notes from a NoteSequence, dropping zero-length notes."""
    ns_trimmed = note_seq.NoteSequence()
    ns_trimmed.CopyFrom(ns)

    # Create a mapping from error class to notes.
    error_class_notes = {}
    for note in ns_trimmed.notes:
        if note.instrument not in error_class_notes:
            error_class_notes[note.instrument] = []
        error_class_notes[note.instrument].append(note)

    # Trim overlapping notes within each error class separately.
    for error_class, notes in error_class_notes.items():
        sorted_notes = sorted(notes, key=lambda note: note.start_time)
        for i in range(1, len(sorted_notes)):
            if sorted_notes[i - 1].end_time > sorted_notes[i].start_time:
                sorted_notes[i - 1].end_time = sorted_notes[i].start_time

    # Compile all notes, ensuring they have a valid duration.
    valid_notes = [
        note
        for note_list in error_class_notes.values()
        for note in note_list
        if note.start_time < note.end_time
    ]
    del ns_trimmed.notes[:]
    ns_trimmed.notes.extend(valid_notes)
    return ns_trimmed


# might be non-useful
def assign_error_classes(ns: note_seq.NoteSequence) -> None:
    """Assign error class to notes; modifies NoteSequence in place."""
    error_class_instruments = {}
    for note in ns.notes:
        if note.instrument not in error_class_instruments:
            error_class_instruments[note.instrument] = note.instrument
        note.instrument = error_class_instruments[note.instrument]


def validate_note_sequence(ns: note_seq.NoteSequence) -> None:
    """Raise ValueError if NoteSequence contains invalid notes."""
    for note in ns.notes:
        if note.start_time >= note.end_time:
            raise ValueError(
                f"note has start time >= end time: {note.start_time} >= {note.end_time}"
            )
        if note.velocity == 0:
            raise ValueError("note has zero velocity")


def note_arrays_to_note_sequence(
    onset_times: Sequence[float],
    pitches: Sequence[int],
    offset_times: Optional[Sequence[float]] = None,
    velocities: Optional[Sequence[int]] = None,
    error_classes: Optional[Sequence[int]] = None,
) -> note_seq.NoteSequence:
    """Convert note onset / offset / pitch / velocity arrays to NoteSequence.
    Each note is associated with an error class.
    """
    ns = note_seq.NoteSequence(ticks_per_quarter=220)
    for onset_time, offset_time, pitch, velocity, error_class in itertools.zip_longest(
        onset_times,
        [] if offset_times is None else offset_times,
        pitches,
        [] if velocities is None else velocities,
        [] if error_classes is None else error_classes,
    ):
        if offset_time is None:
            offset_time = onset_time + DEFAULT_NOTE_DURATION
        if velocity is None:
            velocity = DEFAULT_VELOCITY
        if error_class is None:
            error_class = 0  # Default error class if not specified

        new_note = ns.notes.add()
        new_note.start_time = onset_time
        new_note.end_time = offset_time
        new_note.pitch = pitch
        new_note.velocity = velocity
        new_note.instrument = error_class  # Use 'instrument' field for error class

        ns.total_time = max(ns.total_time, offset_time)
    # Ensure all notes are assigned with the correct error class
    assign_error_classes(ns)
    return ns


@dataclasses.dataclass
class NoteEventData:
    pitch: int
    velocity: Optional[int] = None
    error_class: Optional[int] = None


def note_sequence_to_onsets(
    ns: note_seq.NoteSequence,
) -> Tuple[Sequence[float], Sequence[int]]:
    """Extract note onsets and pitches from NoteSequence proto."""
    notes = sorted(ns.notes, key=lambda note: note.pitch)
    return (
        [note.start_time for note in notes],
        [note.pitch for note in notes],
    )


def note_sequence_to_onsets_and_offsets(
    ns: note_seq.NoteSequence,
) -> Tuple[Sequence[float], Sequence[NoteEventData]]:
    """Extract onset & offset times and pitches from a NoteSequence proto.

    The onset & offset times will not necessarily be in sorted order.

    Args:
      ns: NoteSequence from which to extract onsets and offsets.

    Returns:
      times: A list of note onset and offset times.
      values: A list of NoteEventData objects where velocity is zero for note
          offsets.
    """
    notes = sorted(ns.notes, key=lambda note: note.pitch)
    times = [note.end_time for note in notes] + [note.start_time for note in notes]
    values = [NoteEventData(pitch=note.pitch, velocity=0) for note in notes] + [
        NoteEventData(
            pitch=note.pitch, velocity=note.velocity, error_class=note.instrument
        )
        for note in notes
    ]
    return times, values


# TODO: Rename this function to note_sequence_to_onsets_and_offsets_and_error_classes
def note_sequence_to_onsets_and_offsets_and_programs(
    ns: note_seq.NoteSequence,
) -> Tuple[Sequence[float], Sequence[NoteEventData]]:
    """Extract onset & offset times and pitches & error classes from a NoteSequence.

    The onset & offset times will not necessarily be in sorted order.

    Args:
      ns: NoteSequence from which to extract onsets and offsets.

    Returns:
      times: A list of note onset and offset times.
      values: A list of NoteEventData objects where velocity is zero for note
          offsets, and includes error classes.
    """
    # Sort by error class and pitch and put offsets before onsets as a tiebreaker for
    # subsequent stable sort.
    notes = sorted(ns.notes, key=lambda note: (note.instrument, note.pitch))
    times = [note.end_time for note in notes] + [note.start_time for note in notes]
    values = [
        NoteEventData(pitch=note.pitch, velocity=0, error_class=note.instrument)
        for note in notes
    ] + [
        NoteEventData(
            pitch=note.pitch, velocity=note.velocity, error_class=note.instrument
        )
        for note in notes
    ]
    return times, values


@dataclasses.dataclass
class NoteEncodingState:
    """Encoding state for note transcription, keeping track of active pitches."""

    # velocity bin for active pitches and programs
    active_pitches: MutableMapping[Tuple[int, int], int] = dataclasses.field(
        default_factory=dict
    )


def note_event_data_to_events(
    state: Optional[NoteEncodingState],
    value: NoteEventData,
    codec: event_codec.Codec,
) -> Sequence[event_codec.Event]:
    """Convert note event data to a sequence of events, including error class as part of the state."""
    events = []
    if value.velocity is None:
        # Handle onsets only; no velocity or program
        events.append(event_codec.Event("pitch", value.pitch))
    else:
        # Convert velocity to a bin as per the number of velocity bins in the codec
        num_velocity_bins = vocabularies.num_velocity_bins_from_codec(codec)
        velocity_bin = vocabularies.velocity_to_bin(value.velocity, num_velocity_bins)

        # Error class is handled similarly to program
        if value.error_class is not None:
            if state is not None:
                # Update state to include current error class with pitch and velocity
                state.active_pitches[(value.pitch, value.error_class)] = velocity_bin
            events.extend(
                [
                    event_codec.Event("error_class", value.error_class),
                    event_codec.Event("velocity", velocity_bin),
                    event_codec.Event("pitch", value.pitch),
                ]
            )
        else:
            # No error class, handle as regular velocity and pitch
            if state is not None:
                state.active_pitches[(value.pitch, 0)] = velocity_bin
            events.extend(
                [
                    event_codec.Event("velocity", velocity_bin),
                    event_codec.Event("pitch", value.pitch),
                ]
            )

    return events


def note_encoding_state_to_events(
    state: NoteEncodingState,
) -> Sequence[event_codec.Event]:
    """Output error class and pitch events for active notes plus a final tie event."""
    events = []
    for pitch, error_class in sorted(
        state.active_pitches.keys(), key=lambda k: k[::-1]
    ):
        if state.active_pitches[(pitch, error_class)]:
            events += [
                event_codec.Event("error_class", error_class),  # changed from "program"
                event_codec.Event("pitch", pitch),
            ]
    events.append(event_codec.Event("tie", 0))
    return events


@dataclasses.dataclass
class NoteDecodingState:
    """Decoding state for note transcription."""

    current_time: float = 0.0
    current_velocity: int = DEFAULT_VELOCITY
    current_error_class: int = 0  # Changed from current_program
    active_pitches: MutableMapping[Tuple[int, int], Tuple[float, int]] = (
        dataclasses.field(default_factory=dict)
    )
    tied_pitches: MutableSet[Tuple[int, int]] = dataclasses.field(default_factory=set)
    is_tie_section: bool = False
    note_sequence: note_seq.NoteSequence = dataclasses.field(
        default_factory=lambda: note_seq.NoteSequence(ticks_per_quarter=220)
    )


def decode_note_onset_event(
    state: NoteDecodingState,
    time: float,
    event: event_codec.Event,
    codec: event_codec.Codec,
) -> None:
    """Process note onset event and update decoding state."""
    if event.type == "pitch":
        new_note = state.note_sequence.notes.add()
        new_note.start_time = time
        new_note.end_time = time + DEFAULT_NOTE_DURATION
        new_note.pitch = event.value
        new_note.velocity = DEFAULT_VELOCITY
        new_note.instrument = state.current_error_class
        state.note_sequence.total_time = max(state.note_sequence.total_time, new_note.end_time)
    
    else:
        raise ValueError(f"unexpected event type: {event.type}")


def _add_note_to_sequence(
    ns: note_seq.NoteSequence,
    start_time: float,
    end_time: float,
    pitch: int,
    velocity: int,
    error_class: int = 0,  # Rename 'program' to 'error_class' for clarity
) -> None:
    end_time = max(end_time, start_time + MIN_NOTE_DURATION)
    new_note = ns.notes.add()
    new_note.start_time = start_time
    new_note.end_time = end_time
    new_note.pitch = pitch
    new_note.velocity = velocity
    new_note.instrument = error_class
    ns.total_time = max(ns.total_time, end_time)



def decode_note_event(
    state: NoteDecodingState,
    time: float,
    event: event_codec.Event,
    codec: event_codec.Codec
) -> None:
    """Process note event and update decoding state."""
    if time < state.current_time:
        raise ValueError('event time < current time, %f < %f' % (
            time, state.current_time))
    state.current_time = time
    if event.type == 'pitch':
        pitch = event.value
        if state.is_tie_section:
            # "tied" pitch
            if (pitch, state.current_error_class) not in state.active_pitches:
                raise ValueError('inactive pitch/error_class in tie section: %d/%d' %
                                 (pitch, state.current_error_class))
            if (pitch, state.current_error_class) in state.tied_pitches:
                raise ValueError('pitch/error_class is already tied: %d/%d' %
                                 (pitch, state.current_error_class))
            state.tied_pitches.add((pitch, state.current_error_class))
        elif state.current_velocity == 0:
            # note offset
            if (pitch, state.current_error_class) not in state.active_pitches:
                raise ValueError('note-off for inactive pitch/error_class: %d/%d' %
                                 (pitch, state.current_error_class))
            onset_time, onset_velocity = state.active_pitches.pop(
                (pitch, state.current_error_class))
            _add_note_to_sequence(
                state.note_sequence, start_time=onset_time, end_time=time,
                pitch=pitch, velocity=onset_velocity, error_class=state.current_error_class)
        else:
            # note onset
            if (pitch, state.current_error_class) in state.active_pitches:
                # The pitch is already active; this shouldn't really happen but we'll
                # try to handle it gracefully by ending the previous note and starting a
                # new one.
                onset_time, onset_velocity = state.active_pitches.pop(
                    (pitch, state.current_error_class))
                _add_note_to_sequence(
                    state.note_sequence, start_time=onset_time, end_time=time,
                    pitch=pitch, velocity=onset_velocity, error_class=state.current_error_class)
            state.active_pitches[(pitch, state.current_error_class)] = (
                time, state.current_velocity)
    elif event.type == 'velocity':
        # velocity change
        num_velocity_bins = vocabularies.num_velocity_bins_from_codec(codec)
        velocity = vocabularies.bin_to_velocity(event.value, num_velocity_bins)
        state.current_velocity = velocity
    elif event.type == 'error_class':
        # error_class change
        state.current_error_class = event.value
    elif event.type == 'tie':
        # end of tie section; end active notes that weren't declared tied
        if not state.is_tie_section:
            raise ValueError('tie section end event when not in tie section')
        for (pitch, error_class) in list(state.active_pitches.keys()):
            if (pitch, error_class) not in state.tied_pitches:
                onset_time, onset_velocity = state.active_pitches.pop(
                    (pitch, error_class))
                _add_note_to_sequence(
                    state.note_sequence,
                    start_time=onset_time, end_time=state.current_time,
                    pitch=pitch, velocity=onset_velocity, error_class=error_class)
        state.is_tie_section = False
    else:
        raise ValueError('unexpected event type: %s' % event.type)


def begin_tied_pitches_section(state: NoteDecodingState) -> None:
    """Begin the tied pitches section at the start of a segment."""
    state.tied_pitches = set()
    state.is_tie_section = True


def flush_note_decoding_state(
    state: NoteDecodingState
) -> note_seq.NoteSequence:
    """End all active notes and return resulting NoteSequence."""
    for onset_time, _ in state.active_pitches.values():
        state.current_time = max(
            state.current_time, onset_time + MIN_NOTE_DURATION)
    for (pitch, error_class) in list(state.active_pitches.keys()):
        onset_time, onset_velocity = state.active_pitches.pop((pitch, error_class))
        _add_note_to_sequence(
            state.note_sequence, start_time=onset_time, end_time=state.current_time,
            pitch=pitch, velocity=onset_velocity, error_class=error_class)
    assign_error_classes(state.note_sequence)
    return state.note_sequence

# def flush_note_decoding_state(
#     state: NoteDecodingState
# ) -> note_seq.NoteSequence:
#     """End all active notes and return resulting NoteSequence."""
#     # Step 1: Ensure the current time is updated to cover all notes
#     for onset_time, _ in state.active_pitches.values():
#         state.current_time = max(
#             state.current_time, onset_time + MIN_NOTE_DURATION)

#     # Step 2: End all active notes and add them to the NoteSequence
#     for (pitch, error_class) in list(state.active_pitches.keys()):
#         onset_time, onset_velocity = state.active_pitches.pop((pitch, error_class))
#         _add_note_to_sequence(
#             state.note_sequence, start_time=onset_time, end_time=state.current_time,
#             pitch=pitch, velocity=onset_velocity, error_class=error_class)

#     # Step 3: Ensure all error classes (1, 2, 3) are represented in the final NoteSequence
#     all_error_classes = {1, 2, 3}  # Error classes as defined by EventRange("error_class", 1, 3)
#     present_error_classes = {note.instrument for note in state.note_sequence.notes}
#     missing_error_classes = all_error_classes - present_error_classes

#     # Add an empty note for each missing error class
#     for error_class in missing_error_classes:
#         _add_note_to_sequence(
#             state.note_sequence, start_time=0.0, end_time=0.0,
#             pitch=0, velocity=0, error_class=error_class)

#     # Step 4: Assign error classes to all notes in the sequence
#     assign_error_classes(state.note_sequence)

#     return state.note_sequence


class NoteEncodingSpecType(run_length_encoding.EventEncodingSpec):
    pass


# encoding spec for modeling note onsets only
NoteOnsetEncodingSpec = NoteEncodingSpecType(
    init_encoding_state_fn=lambda: None,
    encode_event_fn=note_event_data_to_events,
    encoding_state_to_events_fn=None,
    init_decoding_state_fn=NoteDecodingState,
    begin_decoding_segment_fn=lambda state: None,
    decode_event_fn=decode_note_onset_event,
    flush_decoding_state_fn=lambda state: state.note_sequence,
)


# encoding spec for modeling onsets and offsets
NoteEncodingSpec = NoteEncodingSpecType(
    init_encoding_state_fn=lambda: None,
    encode_event_fn=note_event_data_to_events,
    encoding_state_to_events_fn=None,
    init_decoding_state_fn=NoteDecodingState,
    begin_decoding_segment_fn=lambda state: None,
    decode_event_fn=decode_note_event,
    flush_decoding_state_fn=flush_note_decoding_state,
)


# encoding spec for modeling onsets and offsets, with a "tie" section at the
# beginning of each segment listing already-active notes
NoteEncodingWithTiesSpec = NoteEncodingSpecType(
    init_encoding_state_fn=NoteEncodingState,
    encode_event_fn=note_event_data_to_events,
    encoding_state_to_events_fn=note_encoding_state_to_events,
    init_decoding_state_fn=NoteDecodingState,
    begin_decoding_segment_fn=begin_tied_pitches_section,
    decode_event_fn=decode_note_event,
    flush_decoding_state_fn=flush_note_decoding_state,
)
