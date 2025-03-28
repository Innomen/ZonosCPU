import json
import torch
import torch.nn as nn
import safetensors
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from zonos.autoencoder import DACAutoencoder
from zonos.backbone import BACKBONES
from zonos.config import ZonosConfig
from zonos.sampling import InferenceParams
from zonos.speaker_cloning import SpeakerEmbeddingLDA
from zonos.utils import DEFAULT_DEVICE

DEFAULT_BACKBONE_CLS = next(iter(BACKBONES.values()))

class Zonos(nn.Module):
    def __init__(self, config: ZonosConfig, backbone_cls=DEFAULT_BACKBONE_CLS):
        super().__init__()
        self.config = config
        self.autoencoder = DACAutoencoder()
        self.backbone = backbone_cls(config.backbone)
        self.prefix_conditioner = config.prefix_conditioner.build(config.d_model)
        self.spk_clone_model = None

        # Embeddings for each codebook
        self.embeddings = nn.ModuleList([nn.Embedding(config.vocab_size, config.d_model) for _ in range(config.n_codebooks)])
        # Heads for each codebook
        self.heads = nn.ModuleList([nn.Linear(config.d_model, config.vocab_size) for _ in range(config.n_codebooks)])

        # Register a hook to pad the embeddings and heads to the correct size
        if config.pad_vocab_to_multiple_of is not None:
            self.register_load_state_dict_post_hook(self._pad_embeddings_and_heads)

    def _pad_embeddings_and_heads(self, *args, **kwargs):
        for w in [*self.embeddings, *self.heads]:
            pad_weight_(w, self.config.pad_vocab_to_multiple_of)

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

        # Load state dict
        sd = {}
        with safetensors.safe_open(model_path, framework="pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
                
        # Fix tensor dimensions before loading
        # Check if dimensions need to be transposed for SpeakerConditioner
        speaker_weight_key = "prefix_conditioner.conditioners.1.project.weight"
        if speaker_weight_key in sd:
            weight = sd[speaker_weight_key]
            if weight.shape == torch.Size([2048, 128]):
                # Original dimensions are correct, no need to transpose
                pass
            elif weight.shape == torch.Size([128, 2048]):
                # Dimensions are reversed, transpose them
                sd[speaker_weight_key] = weight.transpose(0, 1)
        
        # Fix tensor dimensions for EmotionConditioner
        emotion_weight_key = "prefix_conditioner.conditioners.2.weight"
        if emotion_weight_key in sd:
            weight = sd[emotion_weight_key]
            if weight.shape == torch.Size([1024, 8]):
                # Transpose to match our model's dimensions [8, 2048]
                sd[emotion_weight_key] = weight.transpose(0, 1)
        
        # Fix tensor dimensions for FrequencyConditioner
        freq_weight_key = "prefix_conditioner.conditioners.3.weight"
        if freq_weight_key in sd:
            weight = sd[freq_weight_key]
            if weight.shape == torch.Size([1024, 1]):
                # Transpose to match our model's dimensions [1, 2048]
                sd[freq_weight_key] = weight.transpose(0, 1)
        
        # Fix tensor dimensions for PitchConditioner
        pitch_weight_key = "prefix_conditioner.conditioners.4.weight"
        if pitch_weight_key in sd:
            weight = sd[pitch_weight_key]
            if weight.shape == torch.Size([1024, 1]):
                # Transpose to match our model's dimensions [1, 2048]
                sd[pitch_weight_key] = weight.transpose(0, 1)
        
        # Fix tensor dimensions for SpeakingRateConditioner
        rate_weight_key = "prefix_conditioner.conditioners.5.weight"
        if rate_weight_key in sd:
            weight = sd[rate_weight_key]
            if weight.shape == torch.Size([1024, 1]):
                # Transpose to match our model's dimensions [1, 2048]
                sd[rate_weight_key] = weight.transpose(0, 1)
                
        # Handle unexpected keys by removing them
        keys_to_remove = []
        for key in list(sd.keys()):
            if "prefix_conditioner.conditioners.1.uncond_vector" in key:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del sd[key]
                
        # Load state dict with strict=False to ignore missing keys
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
        """Compute logits for the next token."""
        if cfg_scale != 1.0:
            # Classifier-free guidance
            uncond_hidden = self.backbone(
                hidden_states,
                inference_params=inference_params,
                prefix_cond=None,
            )
            cond_hidden = self.backbone(
                hidden_states,
                inference_params=inference_params,
                prefix_cond=inference_params.prefix_cond,
            )
            hidden_states = uncond_hidden + cfg_scale * (cond_hidden - uncond_hidden)
        else:
            hidden_states = self.backbone(
                hidden_states,
                inference_params=inference_params,
                prefix_cond=inference_params.prefix_cond,
            )
        return self.apply_heads(hidden_states)

    def prepare_conditioning(self, cond_dict: dict) -> torch.Tensor:
        """Prepare conditioning for the model."""
        return self.prefix_conditioner(cond_dict)

    def generate(
        self,
        *,
        max_n_codes: int = 1024,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        cfg_scale: float = 1.0,
        callback=None,
        **cond_kwargs,
    ) -> torch.Tensor:
        """Generate audio from conditioning."""
        device = self.device
        cond_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cond_kwargs.items()}
        prefix_cond = self.prepare_conditioning(cond_dict)

        # Initialize with a batch of BOS tokens
        batch_size = prefix_cond.shape[0]
        codes = torch.full((batch_size, 1, self.config.n_codebooks), 0, dtype=torch.long, device=device)

        # Initialize inference parameters
        inference_params = InferenceParams(
            max_n_codes=max_n_codes,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            prefix_cond=prefix_cond,
            callback=callback,
        )

        # Generate codes
        with torch.no_grad():
            for i in tqdm(range(max_n_codes), disable=callback is not None):
                # Embed codes
                hidden_states = self.embed_codes(codes)

                # Compute logits
                logits = self._compute_logits(hidden_states, inference_params, cfg_scale)

                # Sample next token
                next_codes = inference_params.sample(logits)
                codes = torch.cat([codes, next_codes.unsqueeze(1)], dim=1)

                # Update inference parameters
                inference_params.update(i + 1)

                # Call callback if provided
                if callback is not None:
                    callback(i, codes)

                # Check if we're done
                if inference_params.is_done:
                    break

        # Decode codes to audio
        with torch.no_grad():
            audio = self.autoencoder.decode(codes[:, 1:])

        return audio


def pad_weight_(module: nn.Module, pad_to_multiple_of: int) -> None:
    """Pad the weight of a module to a multiple of pad_to_multiple_of."""
    if not hasattr(module, "weight"):
        return

    weight = module.weight
    if weight.size(0) % pad_to_multiple_of == 0:
        return

    pad_size = pad_to_multiple_of - (weight.size(0) % pad_to_multiple_of)
    weight_dtype = weight.dtype
    weight_device = weight.device

    with torch.no_grad():
        module.weight = nn.Parameter(
            torch.cat(
                [
                    weight,
                    torch.zeros(
                        pad_size, *weight.shape[1:], dtype=weight_dtype, device=weight_device
                    ),
                ],
                dim=0,
            )
        )

    if hasattr(module, "bias") and module.bias is not None:
        bias = module.bias
        with torch.no_grad():
            module.bias = nn.Parameter(
                torch.cat(
                    [
                        bias,
                        torch.zeros(pad_size, dtype=bias.dtype, device=bias.device),
                    ],
                    dim=0,
                )
            )
