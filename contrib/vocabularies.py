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

"""Model vocabulary."""

import dataclasses
import math

from typing import Callable, Optional, Sequence
from contrib import event_codec

import note_seq


DECODED_EOS_ID = -1
DECODED_INVALID_ID = -2

# defaults for vocabulary config
DEFAULT_STEPS_PER_SECOND = 100
DEFAULT_MAX_SHIFT_SECONDS = 10
DEFAULT_NUM_VELOCITY_BINS = 127


@dataclasses.dataclass
class VocabularyConfig:
    """Vocabulary configuration parameters."""

    steps_per_second: int = DEFAULT_STEPS_PER_SECOND
    max_shift_seconds: int = DEFAULT_MAX_SHIFT_SECONDS
    num_velocity_bins: int = DEFAULT_NUM_VELOCITY_BINS

    @property
    def abbrev_str(self):
        s = ""
        if self.steps_per_second != DEFAULT_STEPS_PER_SECOND:
            s += "ss%d" % self.steps_per_second
        if self.max_shift_seconds != DEFAULT_MAX_SHIFT_SECONDS:
            s += "ms%d" % self.max_shift_seconds
        if self.num_velocity_bins != DEFAULT_NUM_VELOCITY_BINS:
            s += "vb%d" % self.num_velocity_bins
        return s


def num_velocity_bins_from_codec(codec: event_codec.Codec):
    """Get number of velocity bins from event codec."""
    lo, hi = codec.event_type_range("velocity")
    return hi - lo


def velocity_to_bin(velocity, num_velocity_bins):
    if velocity == 0:
        return 0
    else:
        return math.ceil(num_velocity_bins * velocity / note_seq.MAX_MIDI_VELOCITY)


def bin_to_velocity(velocity_bin, num_velocity_bins):
    if velocity_bin == 0:
        return 0
    else:
        return int(note_seq.MAX_MIDI_VELOCITY * velocity_bin / num_velocity_bins)


def build_codec(vocab_config: VocabularyConfig):
    """Build event codec."""
    event_ranges = [
        event_codec.EventRange(
            "pitch", note_seq.MIN_MIDI_PITCH, note_seq.MAX_MIDI_PITCH
        ),
        event_codec.EventRange("velocity", 0, vocab_config.num_velocity_bins),
        event_codec.EventRange("tie", 0, 0),
        event_codec.EventRange("error_class", 1, 3),  # Defines three error classes.
    ]

    return event_codec.Codec(
        max_shift_steps=(
            vocab_config.steps_per_second * vocab_config.max_shift_seconds
        ),
        steps_per_second=vocab_config.steps_per_second,
        event_ranges=event_ranges,
    )


def vocabulary_from_codec(codec: event_codec.Codec):
    return GenericTokenVocabulary(
        codec.num_classes, extra_ids=100
    )  # we get extra_ids from t5.data.DEFAULT_EXTRA_IDS and remove dependency on t5 library


class GenericTokenVocabulary:
    """Vocabulary with pass-through encoding of tokens."""

    def __init__(self, regular_ids: int, extra_ids: int = 0):
        # The special tokens: 0=PAD, 1=EOS, and 2=UNK (unkown token for rare words)
        self._num_special_tokens = 3
        self._num_regular_tokens = regular_ids
        self.extra_ids = extra_ids
        super().__init__()

    @property
    def eos_id(self) -> Optional[int]:
        return 1

    @property
    def unk_id(self) -> Optional[int]:
        return 2

    @property
    def _base_vocab_size(self) -> int:
        """Number of ids.

        Returns:
          an integer, the vocabulary size
        """
        return self._num_special_tokens + self._num_regular_tokens

    def _encode(self, token_ids: Sequence[int]) -> Sequence[int]:
        """Encode a list of tokens ids as a list of integers."""
        encoded = []
        for token_id in token_ids:
            if not 0 <= token_id < self._num_regular_tokens:
                raise ValueError(
                    f"token_id {token_id} does not fall within valid range of "
                    f"[0, {self._num_regular_tokens})"
                )
            encoded.append(token_id + self._num_special_tokens)
        return encoded

    def _decode(self, ids: Sequence[int]) -> Sequence[int]:
        """Decode a list of integers to a list of token ids."""

        def _decode_id(encoded_id):
            if encoded_id == self.eos_id:
                return DECODED_EOS_ID
            elif encoded_id < self._num_special_tokens:
                return DECODED_INVALID_ID
            elif encoded_id >= self._base_vocab_size:
                return DECODED_INVALID_ID
            else:
                return encoded_id - self._num_special_tokens

        return [_decode_id(int(i)) for i in ids]

    def num_special_tokens(self):
        return self._num_special_tokens

    def __eq__(self, other):
        their_extra_ids = other.extra_ids
        their_num_regular_tokens = other._num_regular_tokens
        return (
            self.extra_ids == their_extra_ids
            and self._num_regular_tokens == their_num_regular_tokens
        )


def num_embeddings(vocabulary: GenericTokenVocabulary) -> int:
    """Vocabulary size as a multiple of 128 for TPU efficiency."""
    return 128 * math.ceil(vocabulary.vocab_size / 128)
