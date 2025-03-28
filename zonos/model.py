import torch
import torch.nn as nn
import json
import safetensors
from typing import Callable
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from zonos.autoencoder import DACAutoencoder
from zonos.backbone import BACKBONES
from zonos.codebook_pattern import apply_delay_pattern, revert_delay_pattern
from zonos.conditioning import PrefixConditioner
from zonos.config import InferenceParams, ZonosConfig
from zonos.sampling import sample_from_logits
from zonos.speaker_cloning import SpeakerEmbeddingLDA
from zonos.utils import DEFAULT_DEVICE, find_multiple, pad_weight_

DEFAULT_BACKBONE_CLS = next(iter(BACKBONES.values()))

class Zonos(nn.Module):
    def __init__(self, config: ZonosConfig, backbone_cls=DEFAULT_BACKBONE_CLS):
        super().__init__()
        self.config = config
        dim = config.backbone.d_model
        self.eos_token_id = config.eos_token_id
        self.masked_token_id = config.masked_token_id

        self.autoencoder = DACAutoencoder()
        self.backbone = backbone_cls(config.backbone)
        self.prefix_conditioner = PrefixConditioner(config.prefix_conditioner, dim)
        self.spk_clone_model = None

        # TODO: pad to multiple of at least 8
        self.embeddings = nn.ModuleList([nn.Embedding(1026, dim) for _ in range(self.autoencoder.num_codebooks)])
        self.heads = nn.ModuleList([nn.Linear(dim, 1025, bias=False) for _ in range(self.autoencoder.num_codebooks)])

        self._cg_graph = None
        self._cg_batch_size = None
        self._cg_input_ids = None
        self._cg_logits = None
        self._cg_inference_params = None
        self._cg_scale = None

        if config.pad_vocab_to_multiple_of:
            self.register_load_state_dict_post_hook(self._pad_embeddings_and_heads)
            
        # Register a custom hook to fix tensor dimension issues
        self.register_load_state_dict_pre_hook(self._fix_tensor_dimensions)

    def _pad_embeddings_and_heads(self, *args, **kwargs):
        for w in [*self.embeddings, *self.heads]:
            pad_weight_(w, self.config.pad_vocab_to_multiple_of)
            
    def _fix_tensor_dimensions(self, state_dict, prefix, *args, **kwargs):
        """
        Custom hook to fix tensor dimension issues before loading state dict
        """
        # Fix SpeakerConditioner project layer dimensions
        speaker_weight_key = "prefix_conditioner.conditioners.1.project.weight"
        speaker_bias_key = "prefix_conditioner.conditioners.1.project.bias"
        
        if speaker_weight_key in state_dict:
            # Check if dimensions need to be transposed
            weight = state_dict[speaker_weight_key]
            if weight.shape == torch.Size([2048, 128]):
                # Original dimensions are correct, no need to transpose
                pass
            elif weight.shape == torch.Size([128, 2048]):
                # Dimensions are reversed, transpose them
                state_dict[speaker_weight_key] = weight.transpose(0, 1)
            
        # Handle unexpected keys by removing them
        keys_to_remove = []
        for key in state_dict:
            if "prefix_conditioner.conditioners.1.uncond_vector" in key:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del state_dict[key]
            
        return state_dict

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(
        cls, repo_id: str, revision: str | None = None, device: str = DEFAULT_DEVICE, **kwargs
    ) -> "Zonos":
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)
        model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=revision)
        return cls.from_local(config_path, model_path, device, **kwargs)

    @classmethod
    def from_local(
        cls, config_path: str, model_path: str, device: str = DEFAULT_DEVICE, backbone: str | None = None
    ) -> "Zonos":
        config = ZonosConfig.from_dict(json.load(open(config_path)))
        if backbone:
            backbone_cls = BACKBONES[backbone]
        else:
            is_transformer = not bool(config.backbone.ssm_cfg)
            backbone_cls = DEFAULT_BACKBONE_CLS
            # Preferentially route to pure torch backbone for increased performance and lower latency.
            if is_transformer and "torch" in BACKBONES:
                backbone_cls = BACKBONES["torch"]

        # CPU-specific modifications: use float32 instead of bfloat16 for CPU compatibility
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        model = cls(config, backbone_cls).to(device, dtype)
        model.autoencoder.dac.to(device)

        # Load state dict with strict=False to ignore missing keys
        sd = model.state_dict()
        with safetensors.safe_open(model_path, framework="pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
                
        # Apply custom dimension fixing before loading
        model.load_state_dict(sd, strict=False)

        return model

    def make_speaker_embedding(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """Generate a speaker embedding from an audio clip."""
        if self.spk_clone_model is None:
            self.spk_clone_model = SpeakerEmbeddingLDA()
        _, spk_embedding = self.spk_clone_model(wav.to(self.spk_clone_model.device), sr)
        # CPU-specific modification: use float32 instead of bfloat16 for CPU compatibility
        dtype = torch.float32 if self.device.type == "cpu" else torch.bfloat16
        return spk_embedding.unsqueeze(0).to(dtype)

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return sum(emb(codes[:, i]) for i, emb in enumerate(self.embeddings))

    def apply_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(hidden_states) for head in self.heads], dim=1)

    def _compute_logits(
        self, hidden_states: torch.Tensor, inference_params: InferenceParams, cfg_scale: float
    ) -> torch.Tensor:
        """
        Pass `hidden_states` into `backbone` and `multi_head`, applying
        classifier-free guidance if `cfg_scale != 1.0`.
        """
        last_hidden_states = self.backbone(hidden_states, inference_params)[:, -1, :].unsqueeze(1)
        logits = self.apply_heads(last_hidden_states).squeeze(2).float()
        if cfg_scale != 1.0:
            cond_logits, uncond_logits = logits.chunk(2)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        logits[..., 1025:].fill_(-torch.inf)  # ensures padding is ignored
        return logits

    def _decode_one_token(
        self,
        input_ids: torch.Tensor,
        inference_params: InferenceParams,
        cfg_scale: float,
        allow_cudagraphs: bool = True,
    ) -> torch.Tensor:
        """
        Single-step decode. Prepares the hidden states, possibly replicates them
        for CFG, and then delegates to `_compute_logits`.
        
        CPU-specific modification: Skip CUDA graph handling for CPU
        """
        # TODO: support cfg_scale==1
        if cfg_scale == 1.0:
            hidden_states = self.embed_codes(input_ids)
            return self._compute_logits(hidden_states, inference_params, cfg_scale)

        bsz = input_ids.size(0)

        # CPU-specific modification: Always use this path for CPU
        if not allow_cudagraphs or input_ids.device.type != "cuda" or self.device.type == "cpu":
            hidden_states_local = self.embed_codes(input_ids)
            hidden_states_local = hidden_states_local.repeat(2, 1, 1)
            return self._compute_logits(hidden_states_local, inference_params, cfg_scale)

        # CUDA-specific code (skipped on CPU)
        need_capture = (self._cg_graph is None) or (self._cg_batch_size != bsz)

        if need_capture:
            self._cg_graph = None

            self._cg_batch_size = bsz
            self._cg_inference_params = inference_params
            self._cg_scale = cfg_scale

            for _ in range(3):
                hidden_states = self.embed_codes(input_ids)
                hidden_states = hidden_states.repeat(2, 1, 1)  # because cfg != 1.0
                logits = self._compute_logits(hidden_states, inference_params, cfg_scale)

            self._cg_input_ids = input_ids.clone()
            self._cg_logits = torch.empty_like(logits)

            g = torch.cuda.CUDAGraph()

            def capture_region():
                hidden_states_local = self.embed_codes(self._cg_input_ids)
                hidden_states_local = hidden_states_local.repeat(2, 1, 1)
                self._cg_logits = self._compute_logits(hidden_states_local, self._cg_inference_params, self._cg_scale)

            with torch.cuda.graph(g):
                capture_region()

            self._cg_graph = g

        else:
            self._cg_input_ids.copy_(input_ids)

        self._cg_graph.replay()

        return self._cg_logits

    def _prefill(
        self,
        prefix_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        inference_params: InferenceParams,
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        "Prefill" mode: we already have `prefix_hidden_states`, and we want
        to append new embeddings, then compute the logits.
        """
        # Replicate input_ids if CFG is enabled
        if cfg_scale != 1.0:
            input_ids = input_ids.expand(prefix_hidden_states.shape[0], -1, -1)
        hidden_states = torch.cat([prefix_hidden_states, self.embed_codes(input_ids)], dim=1)
        return self._compute_logits(hidden_states, inference_params, cfg_scale)

    def setup_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = None) -> InferenceParams:
        # CPU-specific modification: Use float32 as default for CPU
        if dtype is None:
            dtype = torch.float32 if self.device.type == "cpu" else torch.bfloat16
            
        max_seqlen = find_multiple(max_seqlen, 8)
        key_value_memory_dict = self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
        lengths_per_sample = torch.full((batch_size,), 0, dtype=torch.int32)
        return InferenceParams(max_seqlen, batch_size, 0, 0, key_value_memory_dict, lengths_per_sample)

    def prepare_conditioning(self, cond_dict: dict, uncond_dict: dict | None = None) -> torch.Tensor:
        if uncond_dict is None:
            uncond_dict = {k: cond_dict[k] for k in self.prefix_conditioner.required_keys}
        return torch.cat(
            [
                self.prefix_conditioner(cond_dict),
                self.prefix_conditioner(uncond_dict),
            ]
        )

    def can_use_cudagraphs(self) -> bool:
        # CPU-specific modification: Always return False for CPU
        if self.device.type == "cpu":
            return False
        # Only the mamba-ssm backbone supports CUDA Graphs at the moment
        return self.device.type == "cuda" and "_mamba_ssm" in str(self.backbone.__class__)

    @torch.inference_mode()
    def generate(
        self,
        prefix_conditioning: torch.Tensor,  # [bsz, cond_seq_len, d_model]
        audio_prefix_codes: torch.Tensor | None = None,  # [bsz, 9, prefix_audio_seq_len]
        max_new_tokens: int = 86 * 30,
        cfg_scale: float = 2.0,
        batch_size: int = 1,
        sampling_params: dict = dict(min_p=0.1),
        progress_bar: bool = True,
        disable_torch_compile: bool = False,
        callback: Callable[[torch.Tensor, int, int], bool] | None = None,
    ):
        assert cfg_scale != 1, "TODO: add support for cfg_scale=1"
        prefix_audio_len = 0 if audio_prefix_codes is None else audio_prefix_codes.shape[2]
        device = self.device

        # CPU-specific modification: Disable CUDA graphs and torch.compile for CPU
        cg = False if device.type == "cpu" else self.can_use_cudagraphs()
        decode_one_token = self._decode_one_token
        
        # CPU-specific modification: Disable torch.compile for CPU
        if device.type != "cpu" and not (cg or disable_torch_compile):
            try:
                decode_one_token = torch.compile(decode_one_token, dynamic=True)
            except Exception as e:
                print(f"Warning: Failed to compile with error: {e}")
                print("Continuing without compilation")

        unknown_token = -1
        audio_seq_len = prefix_audio_len + max_new_tokens
        seq_len = prefix_conditioning.shape[1] + audio_seq_len + 9

        with torch.device(device):
            inference_params = self.setup_cache(batch_size=batch_size * 2, max_seqlen=seq_len)
            
            # Initialize codes with audio prefix if provided
            if audio_prefix_codes is not None:
                codes = audio_prefix_codes.to(device)
            else:
                codes = torch.full((batch_size, 9, 0), unknown_token, dtype=torch.long, device=device)
            
            # Generate tokens
            pbar = tqdm(range(max_new_tokens), disable=not progress_bar)
            for i in pbar:
                if i == 0 and audio_prefix_codes is not None:
                    # Skip generation for prefix tokens
                    continue
                
                # Get next token
                logits = decode_one_token(codes, inference_params, cfg_scale, allow_cudagraphs=cg)
                next_token = sample_from_logits(logits, **sampling_params)
                
                # Append new token to codes
                codes = torch.cat([codes, next_token.unsqueeze(2)], dim=2)
                
                # Update progress bar
                if progress_bar:
                    pbar.set_description(f"Generating audio: {i+1}/{max_new_tokens} tokens")
                
                # Call callback if provided
                if callback is not None:
                    stop = callback(codes, i, max_new_tokens)
                    if stop:
                        break
            
            return codes
