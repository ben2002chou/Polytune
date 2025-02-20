
import json
import math
import os
from typing import List
import librosa
import numpy as np
from tqdm import tqdm
from models.polytune import T5ForConditionalGeneration, T5Config
import torch.nn as nn
import torch
from contrib import spectrograms, vocabularies, note_sequences, metrics_utils
import note_seq
import traceback
import matplotlib.pyplot as plt

MIN_LOG_MEL = -12
MAX_LOG_MEL = 5


class InferenceHandler:

    def __init__(
        self,
        model=None,
        weight_path=None,
        device=torch.device("cuda"),
        mel_norm=True,
        contiguous_inference=False,
    ) -> None:
        if model is None:
            pass
        else: 
            self.model = model
            self.device = device
            self.model.to(self.device)

        # NOTE: this is only used for XL models, but might change after we train new versions.
        self.contiguous_inference = contiguous_inference

        self.SAMPLE_RATE = 16000
        self.spectrogram_config = spectrograms.SpectrogramConfig()
        self.codec = vocabularies.build_codec(
            vocab_config=vocabularies.VocabularyConfig(num_velocity_bins=1)
        )
        self.vocab = vocabularies.vocabulary_from_codec(self.codec)
        

        self.mel_norm = mel_norm
        # if "pretrained/mt3.pth" in weight_path:
        #     self.mel_norm = False
        # else:
        #     self.mel_norm = True

    # def _audio_to_frames(self, mistake_audio, score_audio, context_multiplier=2):
    #     """Compute spectrogram frames from audio."""
    #     spectrogram_config = self.spectrogram_config
    #     mistake_frame_size = spectrogram_config.hop_width
    #     mistake_padding = [0, mistake_frame_size - len(mistake_audio) % mistake_frame_size]
    #     mistake_audio = np.pad(mistake_audio, mistake_padding, mode="constant")
    #     print("audio", mistake_audio.shape, "frame_size", mistake_frame_size)
    #     mistake_frames, target_length = spectrograms.split_audio(mistake_audio, spectrogram_config)
    #     mistake_num_frames = len(mistake_audio) // mistake_frame_size
    #     mistake_times = np.arange(mistake_num_frames) / spectrogram_config.frames_per_second
    #     score_frames = spectrograms.split_audio_with_extra_context(score_audio, spectrogram_config, target_length, context_multiplier)
    #     # plot the spectrogram frames
        
    #     return mistake_frames, score_frames, mistake_times
    def _audio_to_frames(self, audio):
        """Compute spectrogram frames from audio."""
        spectrogram_config = self.spectrogram_config
        frame_size = spectrogram_config.hop_width
        padding = [0, frame_size - len(audio) % frame_size]
        audio = np.pad(audio, padding, mode='constant')
        # print('audio', audio.shape, 'frame_size', frame_size)
        frames = spectrograms.split_audio(audio, spectrogram_config)
        num_frames = len(audio) // frame_size
        times = np.arange(num_frames) / spectrogram_config.frames_per_second
        return frames, times

    def _split_token_into_length(self, mistake_frames, score_frames, frame_times, max_length=256):
        """
        Batch 1: [Frame0, Frame1, Frame2, Frame3] (no padding needed)
        Batch 2: [Frame4, Frame5, Frame6, Frame7] (no padding needed)
        Batch 3: [Frame8, Frame9, Pad, Pad]      (2 mistake_frames padded)
        max_length: maximum number of frames in a batch
        """
        assert len(mistake_frames.shape) >= 1
        assert mistake_frames.shape[0] == frame_times.shape[0]
        print("mistake_frames", mistake_frames.shape, "score_frames", score_frames.shape, "frame_times", frame_times.shape)
        
        # find the max frame shape
        max_frame_shape = max(mistake_frames.shape[0], score_frames.shape[0])
        
        # pad the frames to be of equal length
        if mistake_frames.shape[0] < max_frame_shape:
            mistake_frames = np.pad(mistake_frames, ((0, max_frame_shape - mistake_frames.shape[0]), (0, 0)), mode='constant')
        if score_frames.shape[0] < max_frame_shape:
            score_frames = np.pad(score_frames, ((0, max_frame_shape - score_frames.shape[0]), (0, 0)), mode='constant')
        
        num_segment = math.ceil(mistake_frames.shape[0] / max_length)  # Use mistake_frames shape here
        mistake_batches = []
        score_batches = []
        mistake_frame_times_batches = []
        mistake_paddings = []
        score_paddings = []
        
        for i in range(num_segment):
            mistake_batch = np.zeros((max_length, *mistake_frames.shape[1:]))
            frame_times_batch = np.zeros((max_length))
            score_batch = np.zeros((2 * max_length, *score_frames.shape[1:]))
            
            start_idx = i * max_length
            end_idx = min(max_length, mistake_frames.shape[0] - start_idx)  # Calculate end_idx based on mistake_frames
            
            score_start_idx = start_idx - max_length // 2
            score_end_idx = min(max_length * 2, score_frames.shape[0] - score_start_idx)
            
            start_padding = 0
            if score_start_idx < 0:
                start_padding = -score_start_idx
                score_start_idx = 0
            
            mistake_batch[0:end_idx, ...] = mistake_frames[start_idx:start_idx + end_idx, ...]
            score_batch[start_padding:start_padding + score_end_idx, ...] = score_frames[score_start_idx:score_start_idx + score_end_idx - start_padding, ...]
            
            # Adjust frame_times_batch to match mistake_frames segmentation
            frame_times_batch[0:end_idx] = frame_times[start_idx:start_idx + end_idx]
            
            mistake_batches.append(mistake_batch)
            score_batches.append(score_batch)
            mistake_frame_times_batches.append(frame_times_batch)
            mistake_paddings.append(end_idx)
            score_paddings.append(score_end_idx)
        
        print("frame_times")
        return np.stack(mistake_batches, axis=0), np.stack(score_batches, axis=0), np.stack(mistake_frame_times_batches, axis=0), mistake_paddings, score_paddings
    def _compute_spectrograms(self, mistake_inputs, score_inputs):
        mistake_outputs = []
 
        for i in mistake_inputs:

            samples = spectrograms.flatten_frames(
                i,
            )
            i = spectrograms.compute_spectrogram(samples, self.spectrogram_config)
            mistake_outputs.append(i)

        mistake_melspec= np.stack(mistake_outputs, axis=0)

        # add normalization
        # NOTE: for MT3 pretrained weights, we don't do mel_norm
        if self.mel_norm:
            mistake_melspec = np.clip(mistake_melspec, MIN_LOG_MEL, MAX_LOG_MEL)
            mistake_melspec = (mistake_melspec - MIN_LOG_MEL) / (MAX_LOG_MEL - MIN_LOG_MEL)
            
        score_outputs = []

        for i in score_inputs:

            samples = spectrograms.flatten_frames(
                i,
            )
            i = spectrograms.compute_spectrogram(samples, self.spectrogram_config, context_multiplier=2)
            score_outputs.append(i)
        
        score_melspec = np.stack(score_outputs, axis=0)
        
        if self.mel_norm:
            score_melspec = np.clip(score_melspec, MIN_LOG_MEL, MAX_LOG_MEL)
            score_melspec = (score_melspec - MIN_LOG_MEL) / (MAX_LOG_MEL - MIN_LOG_MEL)
        
        return mistake_melspec, score_melspec

    def _preprocess(self, mistake_audio, score_audio):
        mistake_frames, frame_times = self._audio_to_frames(mistake_audio)
        # print("mistake_frames", mistake_frames.shape, "frame_times", frame_times.shape)
        score_frames, _ = self._audio_to_frames(score_audio)
        # print("score_frames", score_frames.shape)
        mistake_frames, score_frames, frame_times, mistake_paddings, score_paddings = self._split_token_into_length(
            mistake_frames, score_frames, frame_times
        )
        mistake_inputs, score_inputs = self._compute_spectrograms(mistake_frames, score_frames)
        # print("mistake_inputs", mistake_inputs.shape, "score_inputs", score_inputs.shape)
        for i, p in enumerate(mistake_paddings):

            mistake_inputs[i, p+1:] = 0 
            
        for i, p in enumerate(score_paddings):

            score_inputs[i, int(p/2)+2:] = 0
            
        return mistake_inputs, score_inputs, frame_times

    def _batching(self, mistake_tensors, score_tensors, frame_times, batch_size=1):
        mistake_batches = []
        score_batches = []
        frame_times_batch = []
        for start_idx in range(0, mistake_tensors.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, mistake_tensors.shape[0])
            mistake_batches.append(mistake_tensors[start_idx:end_idx])
            score_batches.append(score_tensors[start_idx:end_idx])
            frame_times_batch.append(frame_times[start_idx:end_idx])
        return mistake_batches, score_batches, frame_times_batch

    def _get_program_ids(self, valid_programs) -> List[List[int]]:
        min_program_id, max_program_id = self.codec.event_type_range("program")
        total_programs = max_program_id - min_program_id
        invalid_programs = []
        for p in range(total_programs):
            if p not in valid_programs:
                invalid_programs.append(p)
        invalid_programs = [min_program_id + id for id in invalid_programs]
        invalid_programs = self.vocab.encode(invalid_programs)
        return [[p] for p in invalid_programs]

    @torch.no_grad()
    def inference(
        self,
        mistake_audio,
        score_audio,
        audio_path=None,
        outpath=None,
        valid_programs=None,
        num_beams=1,
        batch_size=1,
        max_length=1024,
        verbose=False,
    ):
        try:
            if valid_programs is not None:
                invalid_programs = self._get_program_ids(valid_programs)
            else:
                invalid_programs = None
            # print('preprocessing', audio_path)
            mel_length = 256
            mistake_inputs, score_inputs, frame_times = self._preprocess(mistake_audio, score_audio)
            mistake_inputs = mistake_inputs[:, :mel_length, :]
            score_inputs = score_inputs[:, :mel_length, :]
            mistake_inputs_tensor = torch.from_numpy(mistake_inputs)
            score_inputs_tensor = torch.from_numpy(score_inputs)
            results = []
            mistake_inputs_tensor, score_inputs_tensor, frame_times = self._batching(
                mistake_inputs_tensor, score_inputs_tensor, frame_times, batch_size=batch_size
            )
            print("inferencing", audio_path)

            # if self.contiguous_inference:
            #     inputs_tensor = torch.cat(inputs_tensor, dim=0)
            #     frame_times = [torch.tensor(k) for k in frame_times]
            #     frame_times = torch.cat(frame_times, dim=0)
            #     inputs_tensor = [inputs_tensor]
            #     frame_times = [frame_times]

            # self.model.cuda()
            for idx, (mistake_batch, score_batch) in enumerate(zip(mistake_inputs_tensor, score_inputs_tensor)):
                mistake_batch = mistake_batch.to(self.device)
                score_batch = score_batch.to(self.device)
                # print("mistake_batch", mistake_batch.shape, "score_batch", score_batch.shape, flush=True)

                result = self.model.generate(
                    mistake_inputs=mistake_batch,
                    score_inputs=score_batch,
                    max_length=max_length,
                    num_beams=num_beams,
                    do_sample=False,
                    length_penalty=0.4,
                    eos_token_id=self.model.config.eos_token_id,
                    early_stopping=False,
                    bad_words_ids=invalid_programs,
                    use_cache=False,
                )
                result = self._postprocess_batch(result)
                results.append(result)

            event = self._to_event(results, frame_times)
            if outpath is None:
                filename = audio_path.split("/")[-1].split(".")[0]
                outpath = f"./out/{filename}.mid"
            os.makedirs("/".join(outpath.split("/")[:-1]), exist_ok=True)
            print("saving", outpath)
            # converts note sequence to midi file
            note_seq.note_sequence_to_midi_file(event, outpath)

        except Exception as e:
            traceback.print_exc()

    def _postprocess_batch(self, result):
        after_eos = torch.cumsum(
            (result == self.model.config.eos_token_id).float(), dim=-1
        )
        # minus special token
        result = (
            result - self.vocab.num_special_tokens()
        )  # tokens are offset by special tokes.
        result = torch.where(
            after_eos.bool(), -1, result
        )  # mark tokens after EOS as -1 (invalid token)
        # remove bos (SOS token)
        result = result[:, 1:]
        result = result.cpu().numpy()
        return result

    def _to_event(self, predictions_np: List[np.ndarray], frame_times: np.ndarray):
        predictions = []
        for i, batch in enumerate(predictions_np):
            for j, tokens in enumerate(batch):
                tokens = tokens[
                    : np.argmax(tokens == vocabularies.DECODED_EOS_ID)
                ]  # trim after EOS
                start_time = frame_times[i][j][0]  # get start time of the frame
                start_time -= start_time % (
                    1 / self.codec.steps_per_second
                )  # rounding down time. Why?
                predictions.append(
                    {
                        "est_tokens": tokens,
                        "start_time": start_time,
                        "raw_inputs": [],
                    }  # raw_inputs is empty
                )

        encoding_spec = (
            note_sequences.NoteEncodingWithTiesSpec
        )  # here we use ties to tie seperate event frames together
        result = metrics_utils.event_predictions_to_ns(
            predictions, codec=self.codec, encoding_spec=encoding_spec
        )
        return result["est_ns"]



    
    
def save_frames(mistake_frames, score_frames, input_path, file_prefix='frames'):
    output_dir = os.path.dirname(input_path)
    
    num_frames = mistake_frames.shape[0]  # Assuming all frames have the same number
    
    for i in range(num_frames):
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        
        axs[0].imshow(mistake_frames[i,:,:].T, aspect='auto', origin='lower', cmap='viridis')
        axs[0].set_title('Mistake Frames')
        axs[0].set_xlabel('Frame Index')
        axs[0].set_ylabel('Frequency Bin')

        axs[1].imshow(score_frames[i,:,:].T, aspect='auto', origin='lower', cmap='viridis')
        axs[1].set_title('Score Frames _wo')
        axs[1].set_xlabel('Frame Index')
        axs[1].set_ylabel('Frequency Bin')

        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'{file_prefix}_frames_{i}.png')
        plt.savefig(output_path)
        
        plt.close() 
        
def print_audio_length(audio_frames, frame_rate):
    audio_length_seconds = len(audio_frames) / frame_rate
    print(f"Audio length: {audio_length_seconds} seconds", flush=True)
    
