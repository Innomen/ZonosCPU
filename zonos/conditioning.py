import re
import torch
import inflect
from typing import Dict, List, Optional, Union
from transformers import CLIPTextModel, CLIPTokenizer

# For English-only version, we only support English
supported_language_codes = ["en-us"]

class PrefixConditioner(torch.nn.Module):
    """
    Simplified English-only version of the PrefixConditioner
    Removes dependencies on kanjize, sudachipy, and other language-specific libraries
    """
    
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        
        # English number converter
        self.inflect = inflect.engine()
        
        # Text projection to match dimensions with speaker embeddings
        # CLIP embeddings are 768-dimensional, we need to project to d_model (512)
        self.text_projection = torch.nn.Linear(768, self.d_model)
    
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
    
    def forward(self, cond_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the conditioner
        """
        speaker_embeddings = cond_dict.get("speaker_embedding")
        text_embeddings = cond_dict.get("text_embedding")
        
        # Project text embeddings to match d_model dimension
        text_embeddings = self.text_projection(text_embeddings)
        
        # Combine speaker and text embeddings
        combined_embeddings = torch.cat([speaker_embeddings, text_embeddings], dim=1)
        
        return combined_embeddings

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

def make_cond_dict(text: str, speaker: str, language: str = "en-us") -> Dict[str, torch.Tensor]:
    """
    Create a conditioning dictionary for the model
    Simplified English-only version
    
    Args:
        text: Input text to synthesize
        speaker: Speaker name or path to audio file
        language: Language code (only en-us supported in this version)
    
    Returns:
        Dictionary with conditioning tensors
    """
    # Validate language (only English supported in this version)
    if language != "en-us":
        print(f"Warning: Language '{language}' not supported in English-only version. Using en-us instead.")
        language = "en-us"
    
    # Process text (simplified for English only)
    processed_text = process_text(text)
    
    # Load CLIP model for text embeddings (lazy loading)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    
    # Generate text embeddings
    inputs = tokenizer(processed_text, return_tensors="pt", padding=True, truncation=True)
    text_embedding = text_model(**inputs).last_hidden_state.mean(dim=1)
    
    # Create a dummy speaker embedding (in a real scenario, this would come from a speaker encoder)
    # For simplicity, we're creating a random tensor of the right shape
    speaker_embedding = torch.randn(1, 512)  # Assuming 512-dimensional speaker embeddings
    
    # Create the conditioning dictionary
    cond_dict = {
        "text_embedding": text_embedding,
        "speaker_embedding": speaker_embedding,
        "language": language
    }
    
    return cond_dict
