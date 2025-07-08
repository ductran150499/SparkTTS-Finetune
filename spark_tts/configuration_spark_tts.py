# coding=utf-8
# Copyright 2025 SparkAudio & The HuggingFace Inc. team. All rights reserved.
# ... (License headers remain the same) ...
""" SparkTTS model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from typing import List, Optional # Added typing


logger = logging.get_logger(__name__)

# --- Define Individual Sub-Component Config Classes ---

class SparkTTSMelParamsConfig(PretrainedConfig):
    """Configuration for Mel Spectrogram parameters."""
    model_type = "spark-tts-mel-params"
    def __init__(self, sample_rate=16000, n_fft=1024, win_length=640, hop_length=320,
                 mel_fmin=10, mel_fmax=None, num_mels=128, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.num_mels = num_mels

class SparkTTSEncoderConfig(PretrainedConfig):
    """Configuration for the BiCodec Feature Encoder."""
    model_type = "spark-tts-encoder"
    def __init__(self, input_channels=1024, vocos_dim=384, vocos_intermediate_dim=2048,
                 vocos_num_layers=12, out_channels=1024, sample_ratios=[1, 1], **kwargs):
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.vocos_dim = vocos_dim
        self.vocos_intermediate_dim = vocos_intermediate_dim
        self.vocos_num_layers = vocos_num_layers
        self.out_channels = out_channels
        self.sample_ratios = sample_ratios

class SparkTTSDecoderConfig(PretrainedConfig):
    """Configuration for the BiCodec Wave Generator (Decoder)."""
    model_type = "spark-tts-decoder"
    def __init__(self, input_channel=1024, channels=1536, rates=[8, 5, 4, 2],
                 kernel_sizes=[16, 11, 8, 4], **kwargs):
        super().__init__(**kwargs)
        self.input_channel = input_channel
        self.channels = channels
        self.rates = rates
        self.kernel_sizes = kernel_sizes

class SparkTTSQuantizerConfig(PretrainedConfig):
    """Configuration for the BiCodec Factorized Vector Quantizer."""
    model_type = "spark-tts-quantizer"
    def __init__(self, input_dim=1024, codebook_size=8192, codebook_dim=8,
                 commitment=0.25, codebook_loss_weight=2.0, decay=0.99,
                 threshold_ema_dead_code=0.2, **kwargs):
        # Note: Removed use_l2_normlize as it wasn't in the original class __init__ args
        # Add it back if it's actually used by the FactorizedVectorQuantize class init
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment
        self.codebook_loss_weight = codebook_loss_weight
        self.decay = decay
        self.threshold_ema_dead_code = threshold_ema_dead_code

class SparkTTSSpeakerEncoderConfig(PretrainedConfig):
    """Configuration for the BiCodec Speaker Encoder."""
    model_type = "spark-tts-speaker-encoder"
    def __init__(self, input_dim=128, out_dim=1024, latent_dim=128, token_num=32,
                 fsq_levels=[4, 4, 4, 4, 4, 4], fsq_num_quantizers=1, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.token_num = token_num
        self.fsq_levels = fsq_levels
        self.fsq_num_quantizers = fsq_num_quantizers

class SparkTTSPrenetConfig(PretrainedConfig):
    """Configuration for the BiCodec Prenet."""
    model_type = "spark-tts-prenet"
    def __init__(self, input_channels=1024, vocos_dim=384, vocos_intermediate_dim=2048,
                 vocos_num_layers=12, out_channels=1024, condition_dim=1024,
                 sample_ratios=[1, 1], use_tanh_at_final=False, **kwargs):
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.vocos_dim = vocos_dim
        self.vocos_intermediate_dim = vocos_intermediate_dim
        self.vocos_num_layers = vocos_num_layers
        self.out_channels = out_channels
        self.condition_dim = condition_dim
        self.sample_ratios = sample_ratios
        self.use_tanh_at_final = use_tanh_at_final

class SparkTTSPostnetConfig(PretrainedConfig):
    """Configuration for the BiCodec Postnet."""
    model_type = "spark-tts-postnet"
    def __init__(self, input_channels=1024, vocos_dim=384, vocos_intermediate_dim=2048,
                 vocos_num_layers=6, out_channels=1024, use_tanh_at_final=False, **kwargs):
        # Note: Removed condition_dim as it wasn't in the original config example for postnet
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.vocos_dim = vocos_dim
        self.vocos_intermediate_dim = vocos_intermediate_dim
        self.vocos_num_layers = vocos_num_layers
        self.out_channels = out_channels
        self.use_tanh_at_final = use_tanh_at_final


# --- Define the Intermediate BiCodec Config Class ---

class SparkTTSBiCodecConfig(PretrainedConfig):
    """
    Intermediate configuration class for the BiCodec component within SparkTTS.
    It holds instances of the individual sub-component configurations.
    """
    model_type = "spark-tts-bicodec"
    # Map keys in the 'bicodec_config' dict to their respective classes
    sub_configs = {
        "mel_params": SparkTTSMelParamsConfig,
        "encoder_config": SparkTTSEncoderConfig,
        "decoder_config": SparkTTSDecoderConfig,
        "quantizer_config": SparkTTSQuantizerConfig,
        "speaker_encoder_config": SparkTTSSpeakerEncoderConfig,
        "prenet_config": SparkTTSPrenetConfig,
        "postnet_config": SparkTTSPostnetConfig,
    }

    def __init__(
        self,
        mel_params=None,
        encoder_config=None,
        decoder_config=None,
        quantizer_config=None,
        speaker_encoder_config=None,
        prenet_config=None,
        postnet_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Instantiate sub-configs from dictionaries or use defaults/provided instances
        self.mel_params = self._init_sub_config(mel_params, "mel_params")
        self.encoder_config = self._init_sub_config(encoder_config, "encoder_config")
        self.decoder_config = self._init_sub_config(decoder_config, "decoder_config")
        self.quantizer_config = self._init_sub_config(quantizer_config, "quantizer_config")
        self.speaker_encoder_config = self._init_sub_config(speaker_encoder_config, "speaker_encoder_config")
        self.prenet_config = self._init_sub_config(prenet_config, "prenet_config")
        self.postnet_config = self._init_sub_config(postnet_config, "postnet_config")

    def _init_sub_config(self, config_input, config_key):
        """Helper to initialize sub-configs."""
        config_cls = self.sub_configs[config_key]
        if isinstance(config_input, dict):
            return config_cls(**config_input)
        elif config_input is None:
            return config_cls() # Initialize with defaults
        elif isinstance(config_input, config_cls):
            return config_input # Already an instance
        else:
            raise TypeError(f"Invalid type for {config_key}: {type(config_input)}. Expected dict, None, or {config_cls.__name__}.")


# --- Define the Main SparkTTS Config Class ---

class SparkTTSConfig(PretrainedConfig):
    r"""
    Main configuration class for SparkTTSModel, including nested BiCodec configuration.
    Args:
        llm_model_name_or_path (`str`, *optional*, defaults to `"./LLM"`): Path/ID for LLM.
        bicodec_model_name_or_path (`str`, *optional*, defaults to `"./BiCodec"`): Path/ID for BiCodec checkpoint.
        wav2vec2_model_name_or_path (`str`, *optional*, defaults to `"./wav2vec2-large-xlsr-53"`): Path/ID for Wav2Vec2.
        sample_rate (`int`, *optional*, defaults to 16000): Audio sample rate.
        # ... (other top-level args: highpass_cutoff_freq, latent_hop_length, ref_segment_duration, volume_normalize) ...
        bicodec_config (`dict`, *optional*): Dictionary to initialize `SparkTTSBiCodecConfig`.
        torch_dtype (`str`, *optional*, defaults to `"auto"`): Torch dtype.
        kwargs (*optional*): Dictionary of keyword arguments.
    """
    model_type = "spark-tts"
    # Map the key in config.json to the intermediate BiCodec config class
    sub_configs = {"bicodec_config": SparkTTSBiCodecConfig}
    attribute_map = {"hidden_size": "d_model"} # Example

    def __init__(
        self,
        llm_model_name_or_path="./LLM",
        bicodec_model_name_or_path="./BiCodec",
        wav2vec2_model_name_or_path="./wav2vec2-large-xlsr-53",
        sample_rate=16000,
        highpass_cutoff_freq=40,
        latent_hop_length=320,
        ref_segment_duration=6.0,
        volume_normalize=True,
        bicodec_config=None, # Expects a dictionary or None
        torch_dtype="auto",
        **kwargs,
    ):
        # --- Top-level parameters ---
        self.llm_model_name_or_path = llm_model_name_or_path
        self.bicodec_model_name_or_path = bicodec_model_name_or_path
        self.wav2vec2_model_name_or_path = wav2vec2_model_name_or_path
        self.sample_rate = sample_rate
        self.highpass_cutoff_freq = highpass_cutoff_freq
        self.latent_hop_length = latent_hop_length
        self.ref_segment_duration = ref_segment_duration
        self.volume_normalize = volume_normalize
        self.torch_dtype = torch_dtype

        # --- Nested BiCodec Configuration ---
        # Instantiate the intermediate BiCodec config class, which will handle its own sub-configs
        if isinstance(bicodec_config, dict):
            self.bicodec_config = self.sub_configs["bicodec_config"](**bicodec_config)
        elif bicodec_config is None:
            logger.info("`bicodec_config` not provided. Initializing `SparkTTSBiCodecConfig` with its defaults.")
            self.bicodec_config = self.sub_configs["bicodec_config"]()
        elif isinstance(bicodec_config, self.sub_configs["bicodec_config"]):
             self.bicodec_config = bicodec_config # Use existing instance
        else:
             raise TypeError(f"Invalid type for bicodec_config: {type(bicodec_config)}. Expected dict, None, or SparkTTSBiCodecConfig.")


        # Set processor class and auto_map
        kwargs["processor_class"] = kwargs.get("processor_class", "SparkTTSProcessor")
        kwargs["auto_map"] = kwargs.get("auto_map", {
              "AutoConfig": "configuration_spark_tts.SparkTTSConfig",
              "AutoModel": "modeling_spark_tts.SparkTTSModel",
              "AutoProcessor": "processing_spark_tts.SparkTTSProcessor"
            })
        super().__init__(**kwargs)