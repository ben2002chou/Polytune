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

"""Audio spectrogram functions."""

import dataclasses

# for PyTorch spectrogram
import torch
from torchaudio.transforms import MelSpectrogram, Spectrogram, MelScale, TimeStretch
import librosa
import numpy as np

# this is to suppress a warning from torch melspectrogram
import warnings

warnings.filterwarnings("ignore")


# defaults for spectrogram config
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_HOP_WIDTH = 128
DEFAULT_NUM_MEL_BINS = 512

# fixed constants; add these to SpectrogramConfig before changing
FFT_SIZE = 2048
MEL_LO_HZ = 20.0


@dataclasses.dataclass
class SpectrogramConfig:
    """Spectrogram configuration parameters."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    hop_width: int = DEFAULT_HOP_WIDTH
    n_fft: int = FFT_SIZE
    num_mel_bins: int = DEFAULT_NUM_MEL_BINS

    @property
    def abbrev_str(self):
        s = ""
        if self.sample_rate != DEFAULT_SAMPLE_RATE:
            s += "sr%d" % self.sample_rate
        if self.hop_width != DEFAULT_HOP_WIDTH:
            s += "hw%d" % self.hop_width
        if self.num_mel_bins != DEFAULT_NUM_MEL_BINS:
            s += "mb%d" % self.num_mel_bins
        return s

    @property
    def frames_per_second(self):
        return self.sample_rate / self.hop_width


def split_audio(samples, spectrogram_config):
    """Split audio into frames."""

    # print("split PT")
    if samples.shape[0] % spectrogram_config.hop_width != 0:
        samples = np.pad(
            samples,
            (
                0,
                spectrogram_config.hop_width
                - samples.shape[0] % spectrogram_config.hop_width,
            ),
            "constant",
            constant_values=0,
        )
    return librosa.util.frame(
        samples,
        frame_length=spectrogram_config.hop_width,
        hop_length=spectrogram_config.hop_width,
        axis=-1,
    ).T 


# def split_audio_with_extra_context(samples, spectrogram_config, target_length, multiplier=2):
#     """Split audio into frames, with an option for overlapping frames based on the multiplier flag. Includes custom padding for the first and last frames."""

#     frame_length = spectrogram_config.hop_width
#     hop_length = spectrogram_config.hop_width
#     initial_padding = 0
#     final_padding = 0

#     if multiplier != 1:
#         frame_length = multiplier * spectrogram_config.hop_width  # Frame is twice the original hop_width
#         hop_length = spectrogram_config.hop_width # Overlap remains the same hop_width

#         # Initial padding to start the first frame with an empty half hop_width
#         initial_padding = (multiplier - 1) * spectrogram_config.hop_width // 2
#         final_padding = (multiplier - 1) * spectrogram_config.hop_width // 2
    
    

#     # Adjust the total padding required at the end to make the final frame end with an empty half hop_width
#     total_samples_needed = initial_padding + samples.shape[0] + final_padding
     
#     if target_length is not None:
#         compensating_padding = target_length + spectrogram_config.hop_width - total_samples_needed
#         final_padding += compensating_padding
#         if final_padding < 0:
#             print("Warning: final_padding is negative. This may lead to unexpected results.", flush=True)
        
#     # Apply padding to the samples
#     if final_padding < 0:
#         print("Warning: final_padding is negative. This may lead to unexpected results.")
#         print(f"negative final_padding: {final_padding}")
#         samples = np.pad(
#             samples,
#             (initial_padding, 0),
#             "constant",
#             constant_values=(0,)
#         )
#         # trunctate the samples to match the negative final_padding
#         samples = samples[:target_length + spectrogram_config.hop_width]
        
#     else:
#         print("Padding the samples")
#         print("Initial padding:", initial_padding)
#         print("Final padding:", final_padding)
#         samples = np.pad(
#             samples,
#             (initial_padding, final_padding),
#             "constant",
#             constant_values=(0,)
#         )
#         print("Samples shape after padding:", samples.shape)

#     # Frame the audio data
#     return librosa.util.frame(
#         samples,
#         frame_length=frame_length,
#         hop_length=hop_length,
#         axis=-1
#     ).T

def compute_spectrogram(
    samples,
    spectrogram_config,
    context_multiplier=1,
):
    """
    Compute a mel spectrogram.
    """
    if isinstance(samples, np.ndarray):
        samples = torch.from_numpy(samples).float()  # Convert to float32 tensor
        
    # print("spec PT")
    spectrogram_transform = Spectrogram(
        n_fft=FFT_SIZE,
        hop_length=spectrogram_config.hop_width,
        # pad=0,
        # window_fn=torch.hann_window,
        power=None,
        # normalized=False,
        # center=True,
        # pad_mode="reflect",
    )

    time_stretch = TimeStretch(n_freq=FFT_SIZE // 2 + 1)

    mel_scale = MelScale(
        n_mels=spectrogram_config.num_mel_bins,
        sample_rate=spectrogram_config.sample_rate,
        f_min=MEL_LO_HZ,
        n_stft=FFT_SIZE // 2 + 1,
        mel_scale="htk",
    )
    # print(type(samples), samples.shape, samples.dtype, flush=True)
    # samples = torch.from_numpy(samples).float()
    spectrogram = spectrogram_transform(samples)

    spectrogram = time_stretch(spectrogram, context_multiplier)
    S = mel_scale(torch.abs(spectrogram))
    S[S < 0] = 0
    S = torch.log(S + 1e-6)
    return S.numpy().T


def flatten_frames(frames):
    """Convert frames back into a flat array of samples."""

    return np.reshape(frames, (-1,))


def input_depth(spectrogram_config):
    return spectrogram_config.num_mel_bins
