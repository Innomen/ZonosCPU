import re
import torch
import torch.nn as nn
import inflect
from typing import Dict, List, Optional, Union, Iterable, Any
from transformers import CLIPTextModel, CLIPTokenizer

# For English-only version, we only support English
supported_language_codes = ["en-us"]

# Define a base Conditioner class similar to the original
class Conditioner(nn.Module):
    def __init__(
        self,
        output_dim: int,
        name: str,
        cond_dim: int = None,
        projection: str = "none",
        uncond_type: str = "none",
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.output_dim = output_dim
        self.cond_dim = cond_dim = cond_dim or output_dim

        if projection == "linear":
            self.project = nn.Linear(cond_dim, output_dim)
        elif projection == "mlp":
            self.project = nn.Sequential(
                nn.Linear(cond_dim, output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim),
            )
        else:
            self.project = nn.Identity()

        self.uncond_vector = None
        if uncond_type == "learned":
            self.uncond_vector = nn.Parameter(torch.zeros(output_dim))

    def apply_cond(self, *inputs: Any) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, inputs: tuple[Any, ...] | None) -> torch.Tensor:
        if inputs is None:
            assert self.uncond_vector is not None
            return self.uncond_vector.data.view(1, 1, -1)

        cond = self.apply_cond(*inputs)
        cond = self.project(cond)
        return cond


# Simplified English-only text conditioner
class EnglishTextConditioner(Conditioner):
    def __init__(self, output_dim: int, **kwargs):
        super().__init__(output_dim, name="espeak", **kwargs)
        # Create a simple embedding for text
        self.phoneme_embedder = nn.Embedding(1000, output_dim)  # Simplified embedding
        
        # English number converter
        self.inflect = inflect.engine()

    def normalize_text(self, text: str) -> str:
        """
        Normalize text for English only
        """
        # Convert numbers to words (English only)
        def _replace_number(match):
            number = match.group(0)
            try:
                return " " + self.inflect.number_to_words(number) + " "
            except:
                return " " + number + " "
        
        # Find numbers and convert them to words
        text = re.sub(r'\d+', _replace_number, text)
        
        # Basic normalization
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        
        return text

    def apply_cond(self, texts: List[str], languages: List[str]) -> torch.Tensor:
        """
        Simplified English-only text conditioning
        """
        device = self.phoneme_embedder.weight.device
        
        # Process text (simplified for English only)
        processed_texts = [self.normalize_text(text) for text in texts]
        
        # Create simple token IDs (simplified approach)
        token_ids = []
        for text in processed_texts:
            # Simple character-based tokenization
            chars = list(text)
            # Map characters to IDs (simplified)
            ids = [min(ord(c) % 1000, 999) for c in chars]
            token_ids.append(torch.tensor([ids], device=device))
        
        # Get embeddings
        embeddings = [self.phoneme_embedder(ids) for ids in token_ids]
        
        # Pad to same length
        max_len = max(emb.size(1) for emb in embeddings)
        padded_embeddings = []
        for emb in embeddings:
            if emb.size(1) < max_len:
                padding = torch.zeros(1, max_len - emb.size(1), self.output_dim, device=device)
                padded_embeddings.append(torch.cat([emb, padding], dim=1))
            else:
                padded_embeddings.append(emb)
        
        # Stack embeddings
        return torch.cat(padded_embeddings, dim=0)


# Simplified emotion conditioner
class EmotionConditioner(Conditioner):
    def __init__(self, output_dim: int, **kwargs):
        super().__init__(output_dim, name="emotion", **kwargs)
        self.weight = nn.Parameter(torch.randn(output_dim))
        self.uncond_vector = nn.Parameter(torch.zeros(output_dim))

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.view(1, -1, 1)


# Simplified frequency conditioner
class FrequencyConditioner(Conditioner):
    def __init__(self, output_dim: int, **kwargs):
        super().__init__(output_dim, name="fmax", **kwargs)
        self.weight = nn.Parameter(torch.randn(output_dim))
        self.uncond_vector = nn.Parameter(torch.zeros(output_dim))

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.view(1, -1, 1)


# Simplified pitch conditioner
class PitchConditioner(Conditioner):
    def __init__(self, output_dim: int, **kwargs):
        super().__init__(output_dim, name="pitch_std", **kwargs)
        self.weight = nn.Parameter(torch.randn(output_dim))
        self.uncond_vector = nn.Parameter(torch.zeros(output_dim))

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.view(1, -1, 1)


# Simplified speaking rate conditioner
class SpeakingRateConditioner(Conditioner):
    def __init__(self, output_dim: int, **kwargs):
        super().__init__(output_dim, name="speaking_rate", **kwargs)
        self.weight = nn.Parameter(torch.randn(output_dim))
        self.uncond_vector = nn.Parameter(torch.zeros(output_dim))

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.view(1, -1, 1)


# Simplified language ID conditioner
class LanguageIDConditioner(Conditioner):
    def __init__(self, output_dim: int, **kwargs):
        super().__init__(output_dim, name="language_id", **kwargs)
        self.int_embedder = nn.Embedding(1, output_dim)  # Only English
        self.uncond_vector = nn.Parameter(torch.zeros(output_dim))

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        return self.int_embedder(torch.zeros_like(x.squeeze(-1), dtype=torch.long))


# Simplified speaker conditioner
class SpeakerConditioner(Conditioner):
    def __init__(self, output_dim: int, **kwargs):
        super().__init__(output_dim, name="speaker", projection="linear", **kwargs)

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        return x


# Main PrefixConditioner class
class PrefixConditioner(nn.Module):
    """
    English-only version of the PrefixConditioner
    Maintains compatibility with original model structure while removing non-English dependencies
    """
    
    def __init__(self, config=None, d_model: int = 512):
        super().__init__()
        # Handle both initialization patterns:
        # 1. PrefixConditioner(d_model)
        # 2. PrefixConditioner(config, dim)
        if config is not None and isinstance(config, int):
            # If first arg is an int, it's the old-style d_model
            self.d_model = config
        elif config is not None:
            # If first arg is a config object, use d_model from second arg
            self.d_model = d_model
        else:
            # Default case
            self.d_model = d_model
        
        # Create conditioners similar to original structure
        self.conditioners = nn.ModuleList([
            EnglishTextConditioner(self.d_model),
            SpeakerConditioner(self.d_model),
            EmotionConditioner(self.d_model),
            FrequencyConditioner(self.d_model),
            PitchConditioner(self.d_model),
            SpeakingRateConditioner(self.d_model),
            LanguageIDConditioner(self.d_model),
        ])
        
        # Add normalization and projection layers
        self.norm = nn.LayerNorm(self.d_model)
        self.project = nn.Linear(self.d_model, self.d_model)
        
        # Define required keys for conditioning
        self.required_keys = {"espeak", "speaker", "emotion", "fmax", "pitch_std", "speaking_rate", "language_id"}
    
    def forward(self, cond_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the conditioner
        """
        if not set(cond_dict).issuperset(self.required_keys):
            raise ValueError(f"Missing required keys: {self.required_keys - set(cond_dict)}")
        
        conds = []
        for conditioner in self.conditioners:
            conds.append(conditioner(cond_dict.get(conditioner.name)))
        
        max_bsz = max(map(len, conds))
        assert all(c.shape[0] in (max_bsz, 1) for c in conds)
        conds = [c.expand(max_bsz, -1, -1) for c in conds]
        
        return self.norm(self.project(torch.cat(conds, dim=-2)))


# Simple function to process text for English only
def process_text(text: str) -> str:
    """
    Process text for English only, removing all language-specific processing
    """
    # Basic text cleaning
    text = text.strip()
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text


def make_cond_dict(
    text: str = "Hello, world!",
    speaker: str = None,
    language: str = "en-us",
    emotion: List[float] = [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077],
    fmax: float = 22050.0,
    pitch_std: float = 20.0,
    speaking_rate: float = 15.0,
    unconditional_keys: Iterable[str] = set(),
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Create a conditioning dictionary for the model
    Simplified English-only version
    
    Args:
        text: Input text to synthesize
        speaker: Speaker name or path to audio file
        language: Language code (only en-us supported in this version)
        emotion: Emotion vector
        fmax: Maximum frequency
        pitch_std: Standard deviation for pitch
        speaking_rate: Speaking rate in phonemes per minute
        unconditional_keys: Keys to exclude from conditioning
        device: Device to place tensors on
    
    Returns:
        Dictionary with conditioning tensors
    """
    # Validate language (only English supported in this version)
    if language != "en-us":
        print(f"Warning: Language '{language}' not supported in English-only version. Using en-us instead.")
        language = "en-us"
    
    # Process text (simplified for English only)
    processed_text = process_text(text)
    
    # Create speaker embedding if not provided
    if speaker is None:
        speaker_embedding = torch.randn(1, 512)  # Random speaker embedding
    elif isinstance(speaker, str):
        # This would normally load a speaker embedding from a file
        # For simplicity, we're creating a random tensor
        speaker_embedding = torch.randn(1, 512)
    else:
        speaker_embedding = speaker
    
    # Create the conditioning dictionary
    cond_dict = {
        "espeak": ([processed_text], ["en-us"]),
        "speaker": speaker_embedding,
        "emotion": emotion,
        "fmax": fmax,
        "pitch_std": pitch_std,
        "speaking_rate": speaking_rate,
        "language_id": 0,  # English is the only language
    }
    
    # Remove unconditional keys
    for k in unconditional_keys:
        cond_dict.pop(k, None)
    
    # Convert values to tensors
    for k, v in cond_dict.items():
        if isinstance(v, (float, int, list)):
            v = torch.tensor(v)
        if isinstance(v, torch.Tensor):
            cond_dict[k] = v.view(1, 1, -1).to(device)
        
        if k == "emotion":
            cond_dict[k] /= cond_dict[k].sum(dim=-1)
    
    return cond_dict
