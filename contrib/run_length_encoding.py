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

"""Tools for run length encoding."""

import dataclasses
from typing import Any, Callable, Tuple, Optional, Sequence, TypeVar

from absl import logging
from contrib import event_codec

import numpy as np

Event = event_codec.Event

# These should be type variables, but unfortunately those are incompatible with
# dataclasses.
EventData = Any
EncodingState = Any
DecodingState = Any
DecodeResult = Any

T = TypeVar("T", bound=EventData)
ES = TypeVar("ES", bound=EncodingState)
DS = TypeVar("DS", bound=DecodingState)


@dataclasses.dataclass
class EventEncodingSpec:
    """Spec for encoding events."""

    # initialize encoding state
    init_encoding_state_fn: Callable[
        [], EncodingState
    ]  # get function with no arguments and return EncodingState
    # convert EventData into zero or more events, updating encoding state
    encode_event_fn: Callable[
        [EncodingState, EventData, event_codec.Codec], Sequence[event_codec.Event]
    ]
    # convert encoding state (at beginning of segment) into events
    encoding_state_to_events_fn: Optional[
        Callable[[EncodingState], Sequence[event_codec.Event]]
    ]
    # create empty decoding state
    init_decoding_state_fn: Callable[[], DecodingState]
    # update decoding state when entering new segment
    begin_decoding_segment_fn: Callable[[DecodingState], None]
    # consume time and Event and update decoding state
    decode_event_fn: Callable[
        [DecodingState, float, event_codec.Event, event_codec.Codec], None
    ]
    # flush decoding state into result
    flush_decoding_state_fn: Callable[[DecodingState], DecodeResult]


# This function needs to be modified for Debugging purposes
def get_token_name(token_idx):
    token_idx = int(token_idx)
    if token_idx >= 1001 and token_idx <= 1128:
        return f"pitch_{token_idx - 1001}"
    elif token_idx >= 1129 and token_idx <= 1130:
        return f"velocity_{token_idx - 1129}"
    elif token_idx == 1131:
        return "tie"
    elif token_idx >= 1132 and token_idx <= 1134:
        return (
            f"error_{token_idx - 1132}"  # Adjusted to the new range for 3 error types
        )
    elif token_idx >= 0 and token_idx < 1000:
        return f"shift_{token_idx}"
    else:
        return "invalid_{token_idx}"

    return token


def encode_and_index_events(
    state: ES,
    event_times: Sequence[float],
    event_values: Sequence[T],
    encode_event_fn: Callable[[ES, T, event_codec.Codec], Sequence[event_codec.Event]],
    codec: event_codec.Codec,
    frame_times: Sequence[float],
    encoding_state_to_events_fn: Optional[
        Callable[[ES], Sequence[event_codec.Event]]
    ] = None,
) -> Tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int], Sequence[int]]:
    """Encode a sequence of timed events and index to audio frame times.

    Encodes time shifts as repeated single step shifts for later run length
    encoding.

    Optionally, also encodes a sequence of "state events", keeping track of the
    current encoding state at each audio frame. This can be used e.g. to prepend
    events representing the current state to a targets segment.

    Args:
      state: Initial event encoding state.
      event_times: Sequence of event times.
      event_values: Sequence of event values.
      encode_event_fn: Function that transforms event value into a sequence of one
          or more event_codec.Event objects.
      codec: An event_codec.Codec object that maps Event objects to indices.
      frame_times: Time for every audio frame.
      encoding_state_to_events_fn: Function that transforms encoding state into a
          sequence of one or more event_codec.Event objects.

    Returns:
      events: Encoded events and shifts.
      event_start_indices: Corresponding start event index for every audio frame.
          Note: one event can correspond to multiple audio indices due to sampling
          rate differences. This makes splitting sequences tricky because the same
          event can appear at the end of one sequence and the beginning of
          another.
      event_end_indices: Corresponding end event index for every audio frame. Used
          to ensure when slicing that one chunk ends where the next begins. Should
          always be true that event_end_indices[i] = event_start_indices[i + 1].
      state_events: Encoded "state" events representing the encoding state before
          each event.
      state_event_indices: Corresponding state event index for every audio frame.
    """
    indices = np.argsort(event_times, kind="stable")
    event_steps = [round(event_times[i] * codec.steps_per_second) for i in indices]
    event_values = [event_values[i] for i in indices]

    events = []
    state_events = []
    event_start_indices = []
    state_event_indices = []

    cur_step = 0
    cur_event_idx = 0
    cur_state_event_idx = 0

    def fill_event_start_indices_to_cur_step():
        while (
            len(event_start_indices) < len(frame_times)
            and frame_times[len(event_start_indices)]
            < cur_step / codec.steps_per_second
        ):
            event_start_indices.append(cur_event_idx)
            state_event_indices.append(cur_state_event_idx)

    for event_step, event_value in zip(event_steps, event_values):
        while event_step > cur_step:
            events.append(codec.encode_event(Event(type="shift", value=1)))
            cur_step += 1
            fill_event_start_indices_to_cur_step()
            cur_event_idx = len(events)
            cur_state_event_idx = len(state_events)
        if encoding_state_to_events_fn:
            # Dump state to state events *before* processing the next event, because
            # we want to capture the state prior to the occurrence of the event.
            for e in encoding_state_to_events_fn(state):
                # print('* event_step', event_step, 'event_value', event_value, 'codec', e, codec.encode_event(e))
                state_events.append(codec.encode_event(e))
        for e in encode_event_fn(state, event_value, codec):
            # print('event_step', event_step, 'event_value', event_value, 'codec', e, codec.encode_event(e))
            events.append(codec.encode_event(e))

    # After the last event, continue filling out the event_start_indices array.
    # The inequality is not strict because if our current step lines up exactly
    # with (the start of) an audio frame, we need to add an additional shift event
    # to "cover" that frame.
    while cur_step / codec.steps_per_second <= frame_times[-1]:
        events.append(codec.encode_event(Event(type="shift", value=1)))
        cur_step += 1
        fill_event_start_indices_to_cur_step()
        cur_event_idx = len(events)

    # Now fill in event_end_indices. We need this extra array to make sure that
    # when we slice events, each slice ends exactly where the subsequent slice
    # begins.
    event_end_indices = event_start_indices[1:] + [len(events)]

    events = np.array(events)
    state_events = np.array(state_events)
    event_start_indices = np.array(event_start_indices)
    event_end_indices = np.array(event_end_indices)
    state_event_indices = np.array(state_event_indices)

    # print("events", [get_token_name(k) for k in events], flush=True)

    return (
        events,
        event_start_indices,
        event_end_indices,
        state_events,
        state_event_indices,
    )


def decode_events(
    state: DS,
    tokens: np.ndarray,
    start_time: int,
    max_time: Optional[int],
    codec: event_codec.Codec,
    decode_event_fn: Callable[[DS, float, event_codec.Event, event_codec.Codec], None],
) -> Tuple[int, int]:
    """Decode a series of tokens, maintaining a decoding state object.

    Args:
      state: Decoding state object; will be modified in-place.
      tokens: event tokens to convert.
      start_time: offset start time if decoding in the middle of a sequence.
      max_time: Events at or beyond this time will be dropped.
      codec: An event_codec.Codec object that maps indices to Event objects.
      decode_event_fn: Function that consumes an Event (and the current time) and
          updates the decoding state.

    Returns:
      invalid_events: number of events that could not be decoded.
      dropped_events: number of events dropped due to max_time restriction.
    """
    invalid_events = 0
    dropped_events = 0
    cur_steps = 0
    cur_time = start_time
    token_idx = 0
    # print("decode event")
    for token_idx, token in enumerate(tokens):
        try:
            event = codec.decode_event_index(token)
            # print('token', token, 'event', event)
        except ValueError:
            # print('invalid token!', token)
            invalid_events += 1
            continue
        if event.type == "shift":
            cur_steps += event.value
            cur_time = start_time + cur_steps / codec.steps_per_second
            if max_time and cur_time > max_time:
                dropped_events = len(tokens) - token_idx
                break
        else:
            cur_steps = 0
            try:
                decode_event_fn(state, cur_time, event, codec)
            except ValueError:
                # print('invalid token 2!', token)
                invalid_events += 1
                logging.info(
                    "Got invalid event when decoding event %s at time %f. "
                    "Invalid event counter now at %d.",
                    event,
                    cur_time,
                    invalid_events,
                    exc_info=True,
                )
                continue
    return invalid_events, dropped_events
