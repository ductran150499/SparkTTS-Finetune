# coding=utf-8
# Copyright 2025 SparkAudio & The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for SparkTTS. Combines text tokenization and audio feature extraction/processing.
"""

import os # Needed for save_pretrained
import re # For decoding
import torch
import numpy as np
import soundfile as sf # For audio loading
import soxr # For resampling

from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any

from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding # Return type hint
from transformers.feature_extraction_utils import BatchFeature # Return type hint
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.wav2vec2.feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor
from transformers.utils import logging, PushToHubMixin # Added PushToHubMixin
from numpy.lib.stride_tricks import sliding_window_view
import soxr
import soundfile
import random

# Import custom config if needed for defaults
from .configuration_spark_tts import SparkTTSConfig

logger = logging.get_logger(__name__)


# =============================================================================
# >> START: PASTE CODE FROM sparktts/utils/* HERE <<
# =============================================================================
# IMPORTANT: Utility functions needed for processing (audio loading, token parsing)
# must be defined or imported here.

# --- Paste sparktts/utils/audio.py content here ---

def audio_volume_normalize(audio: np.ndarray, coeff: float = 0.2) -> np.ndarray:
    """
    Normalize the volume of an audio signal.

    Parameters:
        audio (numpy array): Input audio signal array.
        coeff (float): Target coefficient for normalization, default is 0.2.

    Returns:
        numpy array: The volume-normalized audio signal.
    """
    # Sort the absolute values of the audio signal
    temp = np.sort(np.abs(audio))

    # If the maximum value is less than 0.1, scale the array to have a maximum of 0.1
    if temp[-1] < 0.1:
        scaling_factor = max(
            temp[-1], 1e-3
        )  # Prevent division by zero with a small constant
        audio = audio / scaling_factor * 0.1

    # Filter out values less than 0.01 from temp
    temp = temp[temp > 0.01]
    L = temp.shape[0]  # Length of the filtered array

    # If there are fewer than or equal to 10 significant values, return the audio without further processing
    if L <= 10:
        return audio

    # Compute the average of the top 10% to 1% of values in temp
    volume = np.mean(temp[int(0.9 * L) : int(0.99 * L)])

    # Normalize the audio to the target coefficient level, clamping the scale factor between 0.1 and 10
    audio = audio * np.clip(coeff / volume, a_min=0.1, a_max=10)

    # Ensure the maximum absolute value in the audio does not exceed 1
    max_value = np.max(np.abs(audio))
    if max_value > 1:
        audio = audio / max_value

    return audio


def load_audio(
    adfile: Path,
    sampling_rate: int = None,
    length: int = None,
    volume_normalize: bool = False,
    segment_duration: int = None,
) -> np.ndarray:
    r"""Load audio file with target sampling rate and lsength

    Args:
        adfile (Path): path to audio file.
        sampling_rate (int, optional): target sampling rate. Defaults to None.
        length (int, optional): target audio length. Defaults to None.
        volume_normalize (bool, optional): whether perform volume normalization. Defaults to False.
        segment_duration (int): random select a segment with duration of {segment_duration}s.
                                Defualt to None which means the whole audio will be used.

    Returns:
        audio (np.ndarray): audio
    """

    audio, sr = soundfile.read(adfile)
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    if sampling_rate is not None and sr != sampling_rate:
        audio = soxr.resample(audio, sr, sampling_rate, quality="VHQ")
        sr = sampling_rate

    if segment_duration is not None:
        seg_length = int(sr * segment_duration)
        audio = random_select_audio_segment(audio, seg_length)

    # Audio volume normalize
    if volume_normalize:
        audio = audio_volume_normalize(audio)
    # check the audio length
    if length is not None:
        assert abs(audio.shape[0] - length) < 1000
        if audio.shape[0] > length:
            audio = audio[:length]
        else:
            audio = np.pad(audio, (0, int(length - audio.shape[0])))
    return audio


def random_select_audio_segment(audio: np.ndarray, length: int) -> np.ndarray:
    """get an audio segment given the length

    Args:
        audio (np.ndarray):
        length (int): audio length = sampling_rate * duration
    """
    if audio.shape[0] < length:
        audio = np.pad(audio, (0, int(length - audio.shape[0])))
    start_index = random.randint(0, audio.shape[0] - length)
    end_index = int(start_index + length)

    return audio[start_index:end_index]

def get_ref_clip(wav: np.ndarray, config) -> np.ndarray: # Needs access to config attributes
    """Get reference audio clip for speaker embedding."""
    # Make sure config has sample_rate, ref_segment_duration, latent_hop_length
    if not all(hasattr(config, attr) for attr in ['sample_rate', 'ref_segment_duration', 'latent_hop_length']):
        raise AttributeError("Config object missing required attributes for get_ref_clip")
    ref_segment_length = (
        int(config.sample_rate * config.ref_segment_duration)
        // config.latent_hop_length
        * config.latent_hop_length
    )
    wav_length = len(wav)
    if ref_segment_length > wav_length:
        wav = np.tile(wav, ref_segment_length // wav_length + 1)
    return wav[:ref_segment_length]


# --- Paste sparktts/utils/token_parser.py content here ---

TASK_TOKEN_MAP = {
    "vc": "<|task_vc|>",
    "tts": "<|task_tts|>",
    "asr": "<|task_asr|>",
    "s2s": "<|task_s2s|>",
    "t2s": "<|task_t2s|>",
    "understand": "<|task_understand|>",
    "caption": "<|task_cap|>",
    "controllable_tts": "<|task_controllable_tts|>",
    "prompt_tts": "<|task_prompt_tts|>",
    "speech_edit": "<|task_edit|>",
}

LEVELS_MAP = {
    "very_low": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "very_high": 4,
}

LEVELS_MAP_UI = {
    1: 'very_low',
    2: 'low',
    3: 'moderate',
    4: 'high',
    5: 'very_high'
}

GENDER_MAP = {
    "female": 0,
    "male": 1,
}

AGE_MAP = {"Child": 0, "Teenager": 1, "Youth-Adult": 2, "Middle-aged": 3, "Elderly": 4}

EMO_MAP = {
    "UNKNOWN": 0,
    "NEUTRAL": 1,
    "ANGRY": 2,
    "HAPPY": 3,
    "SAD": 4,
    "FEARFUL": 5,
    "DISGUSTED": 6,
    "SURPRISED": 7,
    "SARCASTIC": 8,
    "EXCITED": 9,
    "SLEEPY": 10,
    "CONFUSED": 11,
    "EMPHASIS": 12,
    "LAUGHING": 13,
    "SINGING": 14,
    "WORRIED": 15,
    "WHISPER": 16,
    "ANXIOUS": 17,
    "NO-AGREEMENT": 18,
    "APOLOGETIC": 19,
    "CONCERNED": 20,
    "ENUNCIATED": 21,
    "ASSERTIVE": 22,
    "ENCOURAGING": 23,
    "CONTEMPT": 24,
}


class TokenParser:
    """Turn label to special token"""

    def __init__(self):
        pass

    """Parse the attributes of a person."""

    def __init__(self):
        pass

    @staticmethod
    def age(age: str) -> str:
        """Turn age token."""
        age_id = AGE_MAP[age]
        return f"<|age_{age_id}|>"

    @staticmethod
    def gender(gender: str) -> str:
        """Turn gender token."""
        gender_id = GENDER_MAP[gender]
        return f"<|gender_{gender_id}|>"

    @staticmethod
    def mel_value(mel: int):
        """Turn special token of mel scale pitch."""
        mel = max(0, int(mel))
        mel = min(1000, int(mel))
        return f"<|pitch_value_{mel}|>"

    @staticmethod
    def mel_level(level: str):
        """Turn special token of mel level."""
        level_tag = LEVELS_MAP[level]
        return f"<|pitch_label_{level_tag}|>"

    @staticmethod
    def pitch_var_value(pitch_std: int):
        """Turn special token of pitch_std value."""
        assert isinstance(pitch_std, int)
        pitch_std = max(0, int(pitch_std))
        pitch_std = min(10, int(pitch_std))
        return f"<|pitch_var_value_{pitch_std}|>"

    @staticmethod
    def pitch_var_level(level: str):
        """Turn special token of pitch std level."""
        level_tag = LEVELS_MAP[level]
        return f"<|pitch_var_label_{level_tag}|>"

    @staticmethod
    def loudness_value(loudness: int):
        """Turn special toak of loudness value [0, 30]"""
        assert loudness >= 0
        loudness = max(0, int(loudness))
        loudness = min(30, int(loudness))
        return f"<|loudness_value_{loudness}|>"

    @staticmethod
    def loudness_level(level: str):
        """Turn special token of loudness level."""
        level_tag = LEVELS_MAP[level]
        return f"<|loudness_label_{level_tag}|>"

    @staticmethod
    def speed_value(speed: int):
        """Turn special token of speed value."""
        speed = max(0, int(speed))
        speed = min(10, int(speed))
        return f"<|speed_value_{speed}|>"

    @staticmethod
    def speed_level(level: str):
        """Turn special token of speed level."""
        level_tag = LEVELS_MAP[level]
        return f"<|speed_label_{level_tag}|>"

    @staticmethod
    def task(task: str) -> str:
        """Turn special token of task."""
        assert task in TASK_TOKEN_MAP.keys()

        return TASK_TOKEN_MAP[task]

    @staticmethod
    def emotion(emotion: str):
        emo_id = EMO_MAP[emotion]

        return f"<|emotion_{emo_id}|>"

# =============================================================================
# >> END: PASTE CODE FROM sparktts/utils/* HERE <<
# =============================================================================


class SparkTTSProcessor(ProcessorMixin, PushToHubMixin): # Added PushToHubMixin
    r"""
    Constructs a SparkTTS processor which wraps a text tokenizer and relevant audio processing logic.

    Args:
        tokenizer ([`PreTrainedTokenizer`]):
            An instance of [`PreTrainedTokenizer`]. This handles the text tokenization for the LLM.
        feature_extractor ([`Wav2Vec2FeatureExtractor`]):
            An instance of [`Wav2Vec2FeatureExtractor`]. Although Wav2Vec2 features are extracted
            within the model's `tokenize_audio`, the extractor's configuration (like sampling rate)
            is useful, and it aligns with the ProcessorMixin pattern.
        config ([`SparkTTSConfig`], *optional*):
            An instance of [`SparkTTSConfig`] to access configuration parameters like sample rate.
    """
    attributes = ["tokenizer", "feature_extractor"]
    tokenizer_class = "AutoTokenizer"
    feature_extractor_class = "Wav2Vec2FeatureExtractor" # Keep for consistency

    def __init__(self, tokenizer, feature_extractor, config: Optional[SparkTTSConfig] = None, **kwargs):
        super().__init__(tokenizer=tokenizer, feature_extractor=feature_extractor, **kwargs)
        self.model = None
        self.config = config
        # Set sampling rate
        if config and hasattr(config, 'sample_rate'):
             self.sampling_rate = config.sample_rate
        elif feature_extractor and hasattr(feature_extractor, 'sampling_rate'):
             self.sampling_rate = feature_extractor.sampling_rate
        else:
             self.sampling_rate = 16000
             logger.warning(f"Could not determine sampling rate. Defaulting to {self.sampling_rate} Hz.")

        # # Ensure tokenizer pad token
        # if self.tokenizer.pad_token is None:
        #     if self.tokenizer.eos_token is not None:
        #         logger.warning("Tokenizer does not have a pad token. Setting pad_token to eos_token.")
        #         self.tokenizer.pad_token = self.tokenizer.eos_token
        #     else:
        #          logger.warning("Tokenizer lacks pad and eos token. Adding default pad token '<|pad|>'.")
        #          self.tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    def link_model(self, model):
        """Links the processor to a SparkTTSModel instance for audio processing calls."""
        if not hasattr(model, 'tokenize_audio') or not hasattr(model, 'detokenize_audio'):
             raise TypeError("The provided model instance does not have the required 'tokenize_audio' and 'detokenize_audio' methods.")
        if not hasattr(model, 'config'):
             logger.warning("Linked model does not have a 'config' attribute. Some processor functionalities might rely on it.")

        self.model = model
        logger.info("SparkTTSModel successfully linked to the processor.")
        # Update sampling rate based on linked model's config if available
        if hasattr(model, 'config') and hasattr(model.config, 'sample_rate'):
             if self.sampling_rate != model.config.sample_rate:
                  logger.info(f"Updating processor sampling rate from {self.sampling_rate} to {model.config.sample_rate} based on linked model config.")
                  self.sampling_rate = model.config.sample_rate
             # Also update feature extractor sampling rate if it differs
             if hasattr(self, 'feature_extractor') and self.feature_extractor.sampling_rate != model.config.sample_rate:
                  logger.info(f"Updating feature_extractor sampling rate from {self.feature_extractor.sampling_rate} to {model.config.sample_rate}.")
                  self.feature_extractor.sampling_rate = model.config.sample_rate


    def __call__(
        self,
        text: str,
        prompt_speech_path: Optional[Union[str, Path]] = None,
        prompt_text: Optional[str] = None,
        gender: Optional[str] = None,
        pitch: Optional[str] = None,
        speed: Optional[str] = None,
        return_tensors: Optional[str] = "pt",
        **kwargs, # Allow passing other args like padding, truncation to tokenizer
    ) -> BatchEncoding:
        """
        Processes the input text and optional prompt audio/control parameters into a format suitable for [`SparkTTSModel`].

        Args:
            text (`str`):
                The main text to be synthesized.
            prompt_speech_path (`str` or `Path`, *optional*):
                Path to the prompt audio file for voice cloning. Required if `gender` is not set.
            prompt_text (`str`, *optional*):
                Transcript of the prompt audio. Used only in voice cloning mode.
            gender (`str`, *optional*):
                Target gender ("male" or "female") for controllable synthesis. If set, enables control mode.
            pitch (`str`, *optional*):
                Target pitch level ("very_low", "low", "moderate", "high", "very_high") for control mode. Required if `gender` is set.
            speed (`str`, *optional*):
                Target speed level ("very_low", "low", "moderate", "high", "very_high") for control mode. Required if `gender` is set.
            return_tensors (`str`, *optional*, defaults to `"pt"`):
                If set, will return tensors instead of list of python integers. Only "pt" (PyTorch) is supported currently.
            **kwargs:
                Additional arguments passed to the underlying tokenizer's `__call__` method.

        Returns:
            [`BatchEncoding`]: A dictionary containing the `input_ids` and `attention_mask` for the LLM.
            In voice cloning mode, it also includes `global_token_ids_prompt` (torch.Tensor) representing the
            global tokens extracted from the prompt audio.
        """

        global_token_ids_prompt = None # Initialize

        # Determine mode: Control TTS or Voice Cloning (Prompt TTS)
        is_control_mode = gender is not None
        is_cloning_mode = prompt_speech_path is not None and not is_control_mode

        if is_control_mode:
            # --- Controllable TTS Prompt Construction ---
            if not all([pitch, speed]):
                raise ValueError("For controllable TTS, 'gender', 'pitch', and 'speed' must all be provided.")
            if prompt_speech_path is not None:
                 logger.warning("`prompt_speech_path` provided but ignored because `gender` is set (controllable TTS mode).")

            if not all(k in GENDER_MAP for k in [gender]): # Basic check
                 raise ValueError(f"Invalid gender provided: {gender}. Must be one of {list(GENDER_MAP.keys())}")
            if not all(k in LEVELS_MAP for k in [pitch, speed]): # Basic check
                 raise ValueError(f"Invalid pitch or speed level provided. Must be one of {list(LEVELS_MAP.keys())}")

            gender_id = GENDER_MAP[gender]
            pitch_level_id = LEVELS_MAP[pitch]
            speed_level_id = LEVELS_MAP[speed]

            pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
            speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
            gender_tokens = f"<|gender_{gender_id}|>"

            attribute_tokens = "".join([gender_tokens, pitch_label_tokens, speed_label_tokens])

            prompt_list = [
                TASK_TOKEN_MAP["controllable_tts"],
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_style_label|>",
                attribute_tokens,
                "<|end_style_label|>",
            ]
            prompt_string = "".join(prompt_list)

        elif is_cloning_mode:
            # --- Voice Cloning Prompt Construction ---
            if self.model is None:
                raise RuntimeError("Processor must be linked to a SparkTTSModel instance via `processor.link_model(model)` before performing voice cloning.")
            prompt_speech_path = Path(prompt_speech_path) # Ensure it's a Path object
            if not prompt_speech_path.exists():
                 raise FileNotFoundError(f"Prompt audio file not found: {prompt_speech_path}")

            # Load and process prompt audio
            try:
                model_config = self.model.config if self.model and hasattr(self.model, 'config') else self.config
                if model_config is None:
                     raise ValueError("Configuration not available in processor or linked model.")

                # Load main wav
                wav = load_audio(
                    prompt_speech_path,
                    sampling_rate=self.sampling_rate,
                    volume_normalize=getattr(model_config, 'volume_normalize', True), # Use getattr for safety
                )
                # Get reference clip
                wav_ref_np = get_ref_clip(wav, model_config) # Pass config object
                wav_ref = torch.from_numpy(wav_ref_np).unsqueeze(0).float()
                wav_tensor = torch.from_numpy(wav).unsqueeze(0).float()

                # Tokenize using the linked model's method
                # Assuming tokenize_audio returns tensors with batch dim 1: [1, N_global], [1, N_semantic]
                global_tokens_tensor, semantic_tokens_tensor = self.model.tokenize_audio(wav_tensor, wav_ref)

                # Store the global tokens tensor (with batch dim) for the output dict
                global_token_ids_prompt = global_tokens_tensor # Keep batch dim [1, N_global]

                # Convert tensors to lists of ints for string formatting
                global_token_list = global_tokens_tensor.squeeze().tolist() # Remove batch dim -> list
                semantic_token_list = semantic_tokens_tensor.squeeze().tolist() # Remove batch dim -> list

            except Exception as e:
                logger.error(f"Error processing prompt audio {prompt_speech_path}: {e}")
                import traceback
                traceback.print_exc()
                raise

            # ==============================================================
            # CORRECTED TOKEN STRING FORMATTING
            # ==============================================================
            # Create individual token strings for each ID
            global_tokens_str = "".join([f"<|bicodec_global_{gid}|>" for gid in global_token_list])
            semantic_tokens_str = "".join([f"<|bicodec_semantic_{sid}|>" for sid in semantic_token_list])
            # ==============================================================
            
            # Construct prompt list based on presence of prompt_text
            if prompt_text is not None and prompt_text.strip(): # Check if prompt_text is meaningful
                logger.info("Using prompt text in voice cloning prompt.")
                prompt_list = [
                    TASK_TOKEN_MAP["tts"], # Or maybe TASK_TOKEN_MAP["prompt_tts"]? Check original logic. Assuming "tts".
                    "<|start_content|>",
                    prompt_text, # Transcript first
                    text,        # Then target text
                    "<|end_content|>",
                    "<|start_global_token|>",
                    global_tokens_str,
                    "<|end_global_token|>",
                    "<|start_semantic_token|>",
                    semantic_tokens_str,
                    # "<|end_semantic_token|>", # Original code didn't have this marker here
                ]
            else:
                 # Simpler prompt without semantic tokens if no transcript provided
                 logger.info("No prompt text provided, using text-only voice cloning prompt.")
                 prompt_list = [
                    TASK_TOKEN_MAP["tts"], # Or maybe TASK_TOKEN_MAP["prompt_tts"]?
                    "<|start_content|>",
                    text, # Only target text
                    "<|end_content|>",
                    "<|start_global_token|>",
                    global_tokens_str,
                    "<|end_global_token|>",
                 ]
            prompt_string = "".join(prompt_list)
            logger.debug(f"Generated prompt string (cloning): {prompt_string[:200]}...") # Log start of prompt

        else:
            raise ValueError("Invalid input combination. Either provide `prompt_speech_path` for cloning or (`gender`, `pitch`, `speed`) for control.")

        # --- Tokenize the final prompt string ---
        # print(f"Tokenizing prompt: {prompt_string}")
        inputs = self.tokenizer(
            prompt_string,
            return_tensors=return_tensors,
            padding=kwargs.get("padding", False), # Often False for generation prompts unless batching > 1
            truncation=kwargs.get("truncation", True),
            max_length=kwargs.get("max_length", self.tokenizer.model_max_length),
            add_special_tokens=kwargs.get("add_special_tokens", True), # Usually True unless handled manually
            return_attention_mask=kwargs.get("return_attention_mask", True), # Need attention mask
            **{k: v for k, v in kwargs.items() if k not in ["padding", "truncation", "max_length", "add_special_tokens", "return_attention_mask"]}
        )
        logger.debug(f"Tokenized input_ids shape: {inputs['input_ids'].shape}")


        # Add the prompt's global tokens (as tensor with batch dim) to the output if in cloning mode
        if is_cloning_mode and global_token_ids_prompt is not None:
            if return_tensors == "pt":
                 inputs["global_token_ids_prompt"] = global_token_ids_prompt # Already has batch dim [1, N_global]
            else:
                 # Handle non-tensor return if necessary
                 inputs["global_token_ids_prompt"] = global_token_ids_prompt.tolist()

        return inputs


    def decode(
        self,
        generated_ids: torch.Tensor,
        global_token_ids_prompt: Optional[torch.Tensor] = None,
        input_ids_len: Optional[int] = None,
        skip_special_tokens: bool = True,
    ) -> Dict[str, Any]:
        """
        Decodes the generated token IDs from [`SparkTTSModel`] into an audio waveform.

        Args:
            generated_ids (`torch.Tensor`):
                Tensor of token IDs generated by `model.generate()`, including the input prompt part. Shape [B, seq_len].
            global_token_ids_prompt (`torch.Tensor`, *optional*):
                The global tokens extracted from the prompt audio during the `__call__` step (for voice cloning).
                Shape [B, N_global]. Required if the generation was for voice cloning.
            input_ids_len (`int`, *optional*):
                The length of the original input prompt `input_ids` fed to `model.generate()`. Required to
                correctly isolate the newly generated tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether to skip special tokens during the text decoding step (used to extract audio tokens).

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "audio": The decoded audio waveform as a NumPy array. Shape [T_audio] (if B=1) or [B, T_audio].
                - "sampling_rate": The sampling rate of the audio.
        """
        if self.model is None:
            raise RuntimeError("Processor must be linked to a SparkTTSModel instance via `processor.link_model(model)` before decoding.")
        if input_ids_len is None:
            raise ValueError("`input_ids_len` (length of the prompt input_ids) must be provided for decoding.")

        # --- Isolate generated part and decode text ---
        # Assumes generated_ids has shape [B, full_seq_len]
        # Handle case where generated sequence is shorter than prompt (shouldn't happen with max_new_tokens > 0)
        if generated_ids.shape[1] < input_ids_len:
             logger.warning(f"Generated sequence length ({generated_ids.shape[1]}) is shorter than input prompt length ({input_ids_len}). Decoding might be incorrect.")
             output_only_ids = generated_ids[:, input_ids_len:] # Will be empty if equal
        else:
             output_only_ids = generated_ids[:, input_ids_len:]


        # Decode the generated part to find audio tokens
        # Need to handle batch decoding if B > 1
        # print("decode token", self.tokenizer.batch_decode(output_only_ids, skip_special_tokens=False))
        decoded_texts = self.tokenizer.batch_decode(output_only_ids, skip_special_tokens=skip_special_tokens)

        # --- Extract Audio Tokens ---
        # Handle batch processing correctly
        batch_size = generated_ids.shape[0]
        all_semantic_ids = []
        all_global_tokens = []
        successful_indices = [] # Keep track of which batch items were successful

        for i in range(batch_size):
            decoded_text = decoded_texts[i]
            current_semantic_ids = None
            current_global_tokens = None

            # Extract semantic tokens
            try:
                pred_semantic_indices = [int(token) for token in re.findall(r"bicodec_semantic_(\d+)", decoded_text)]
                if not pred_semantic_indices:
                    logger.warning(f"Batch item {i}: No semantic tokens found in decoded text: '{decoded_text[:200]}...'")
                    continue # Skip this item

                current_semantic_ids = torch.tensor(pred_semantic_indices).long() # Shape [N_semantic]
            except Exception as e:
                logger.error(f"Batch item {i}: Error parsing semantic tokens from: '{decoded_text[:200]}...'. Error: {e}")
                continue # Skip this item

            # Determine global tokens
            if global_token_ids_prompt is not None:
                # Cloning mode: Use the provided prompt global tokens for this batch item
                if global_token_ids_prompt.shape[0] != batch_size:
                     raise ValueError(f"Batch size mismatch: generated_ids has {batch_size}, but global_token_ids_prompt has {global_token_ids_prompt.shape[0]}.")
                current_global_tokens = global_token_ids_prompt[i] # Shape [N_global]
            else:
                # Control mode: Extract global tokens from the generated text
                try:
                    pred_global_indices = [int(token) for token in re.findall(r"bicodec_global_(\d+)", decoded_text)]
                    if not pred_global_indices:
                        logger.warning(f"Batch item {i}: No global tokens found in decoded text for control mode: '{decoded_text[:200]}...'")
                        continue # Skip this item

                    current_global_tokens = torch.tensor(pred_global_indices).long() # Shape [N_global]

                except Exception as e:
                    logger.error(f"Batch item {i}: Error parsing global tokens from: '{decoded_text[:200]}...'. Error: {e}")
                    continue # Skip this item

            # If both tokens extracted successfully
            all_semantic_ids.append(current_semantic_ids)
            all_global_tokens.append(current_global_tokens)
            successful_indices.append(i)

        if not successful_indices:
            logger.error("Failed to extract audio tokens for any item in the batch.")
            return {"audio": np.array([], dtype=np.float32), "sampling_rate": self.sampling_rate}

        # Pad sequences to the max length within the successful batch items for batch detokenization
        # Note: BiCodec might not support batching if sequences have different lengths. Check its implementation.
        # Assuming BiCodec *can* handle batches if padded (or if lengths are naturally equal).
        # This padding might be unnecessary if BiCodec handles variable lengths or if B=1 anyway.
        # For now, let's assume B=1 was handled correctly and skip complex padding.
        if batch_size > 1 and len(successful_indices) < batch_size:
             logger.warning(f"Only successfully decoded {len(successful_indices)} out of {batch_size} batch items.")
             # Further processing might need to handle only the successful items.

        # Let's proceed assuming B=1 or BiCodec handles batches appropriately.
        # Stack the successful tokens.
        try:
            # Need to ensure tensors have the same length before stacking if BiCodec requires it.
            # If BiCodec handles variable length, stacking might not be needed, just loop and call detokenize.
            # Let's assume B=1 for simplicity of the example, matching original code's likely behavior.
            if len(successful_indices) != 1:
                 raise NotImplementedError("Batch decoding (B > 1) requires verification of BiCodec's batch handling and potentially padding.")

            final_semantic_ids = all_semantic_ids[0].unsqueeze(0) # Add batch dim [1, N_semantic]
            final_global_tokens = all_global_tokens[0].unsqueeze(0) # Add batch dim [1, N_global]

        except IndexError: # Should not happen if successful_indices is not empty
             logger.error("Internal error during token batch preparation.")
             return {"audio": np.array([], dtype=np.float32), "sampling_rate": self.sampling_rate}


        # --- Detokenize Audio ---
        try:
            # Call the linked model's detokenize method
            # print(f"DEBUG: Detokenizing audio with global tokens {final_global_tokens.shape}, semantic tokens {final_semantic_ids.shape}")
            output_wav = self.model.detokenize_audio(final_global_tokens, final_semantic_ids)
            # detokenize_audio now returns numpy array float32 in [-1, 1]

            # Optional: Double-check dtype here if needed, but should be handled by detokenize_audio now
            # if output_wav.dtype != np.float32:
            #    logger.warning(f"Audio dtype after detokenize is {output_wav.dtype}. Converting to float32.")
            #    output_wav = output_wav.astype(np.float32)
            # output_wav = np.clip(output_wav, -1.0, 1.0) # Clipping done in detokenize_audio

        except Exception as e:
             logger.error(f"Error during audio detokenization: {e}")
             import traceback
             traceback.print_exc()
             raise RuntimeError("Audio detokenization failed.") from e

        return {"audio": output_wav, "sampling_rate": self.sampling_rate}


    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        trust_remote_code: bool = False, # Allow passing this, needed for config potentially
        **kwargs,
    ):
        r"""
        Instantiate a SparkTTSProcessor from pretrained components.
        """
        # Pop specific kwargs for this method
        config = kwargs.pop("config", None) # Allow passing config explicitly

        # --- 1. Load Config (to find component paths) ---
        # We need the config even if the processor doesn't store it permanently,
        # just to find where the tokenizer/feature_extractor live.
        loaded_config = None
        if not isinstance(config, SparkTTSConfig):
            try:
                # Load the specific config class
                loaded_config = SparkTTSConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    trust_remote_code=trust_remote_code, # Config might be custom
                    **kwargs, # Pass relevant kwargs
                )
            except Exception as e:
                logger.warning(
                    f"Could not load SparkTTSConfig from {pretrained_model_name_or_path}. "
                    f"Attempting to load components from default relative paths ('LLM', 'wav2vec2-large-xlsr-53'). Error: {e}"
                )
                loaded_config = None # Fallback
        else:
             # Config object was passed directly
             loaded_config = config


        # --- 2. Determine Component Paths ---
        llm_tokenizer_path_or_id = "./LLM" # Default relative path
        w2v_processor_path_or_id = "./wav2vec2-large-xlsr-53" # Default relative path

        if loaded_config:
            llm_tokenizer_path_or_id = getattr(loaded_config, 'llm_model_name_or_path', llm_tokenizer_path_or_id)
            w2v_processor_path_or_id = getattr(loaded_config, 'wav2vec2_model_name_or_path', w2v_processor_path_or_id)

        # The component `from_pretrained` methods handle resolving these paths/IDs
        # whether they are relative subfolders of `pretrained_model_name_or_path`
        # or separate Hub IDs.

        # --- 3. Load Components ---
        # Pass down relevant kwargs for loading components
        component_loading_kwargs = {
             "cache_dir": cache_dir,
             "force_download": force_download,
             "local_files_only": local_files_only,
             "token": token,
             "revision": revision,
             **kwargs # Pass other user kwargs
        }
        try:
            # Tokenizer might require trust_remote_code if its class is custom
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, # Main path
                subfolder=llm_tokenizer_path_or_id.lstrip('./'), # Specify subfolder relative to main path
                trust_remote_code=trust_remote_code,
                **component_loading_kwargs
            )
        except Exception as e:
            # Fallback: try loading directly using the path/id from config if different
            if llm_tokenizer_path_or_id != "./LLM":
                 try:
                      logger.info(f"Retrying tokenizer load directly from: {llm_tokenizer_path_or_id}")
                      tokenizer = AutoTokenizer.from_pretrained(
                           llm_tokenizer_path_or_id,
                           trust_remote_code=trust_remote_code,
                           **component_loading_kwargs
                      )
                 except Exception as e2:
                      raise OSError(f"Could not load tokenizer using main path + subfolder or directly from '{llm_tokenizer_path_or_id}'. Error: {e2}") from e
            else:
                 raise OSError(f"Could not load tokenizer from subfolder '{llm_tokenizer_path_or_id}' within '{pretrained_model_name_or_path}'. Error: {e}")


        try:
            # Feature extractor usually doesn't need trust_remote_code
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                pretrained_model_name_or_path, # Main path
                subfolder=w2v_processor_path_or_id.lstrip('./'), # Specify subfolder relative to main path
                **component_loading_kwargs
            )
        except Exception as e:
             # Fallback: try loading directly using the path/id from config if different
            if w2v_processor_path_or_id != "./wav2vec2-large-xlsr-53":
                 try:
                      logger.info(f"Retrying feature extractor load directly from: {w2v_processor_path_or_id}")
                      feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                           w2v_processor_path_or_id,
                           **component_loading_kwargs
                      )
                 except Exception as e2:
                      raise OSError(f"Could not load feature extractor using main path + subfolder or directly from '{w2v_processor_path_or_id}'. Error: {e2}") from e
            else:
                 raise OSError(f"Could not load feature extractor from subfolder '{w2v_processor_path_or_id}' within '{pretrained_model_name_or_path}'. Error: {e}")


        # --- 4. Instantiate processor ---
        # Pass the potentially loaded config object (or None)
        return cls(tokenizer=tokenizer, feature_extractor=feature_extractor, config=loaded_config)

    
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save the processor's state (tokenizer and feature extractor files) to a directory.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the processor files will be saved.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face Hub after saving it.
            **kwargs:
                Additional key word arguments passed along to the `push_to_hub` method.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save tokenizer
        self.tokenizer.save_pretrained(str(save_directory), **kwargs)

        # Save feature extractor
        self.feature_extractor.save_pretrained(str(save_directory), **kwargs)

        # Save the main processor config (if it exists and has relevant info)
        # Note: The SparkTTSConfig is usually saved with the *model*, not the processor.
        # However, if the processor holds specific config needed for reloading *itself*,
        # it could be saved here. Usually, relying on the model's config is sufficient.
        # if self.config:
        #    self.config.save_pretrained(str(save_directory)) # Example if needed

        logger.info(f"Processor components saved in {save_directory}")

        if push_to_hub:
             # Commit message and other hub kwargs can be passed via **kwargs
             commit_message = kwargs.pop("commit_message", "Save processor")
             return self.push_to_hub(save_directory, commit_message=commit_message, **kwargs)

        return str(save_directory) # Return path consistent with Mixin