#!/usr/bin/env python3
"""
Simple script to generate audio using the David Attenborough voice sample
Bypasses the GUI and directly uses the model to generate speech
"""

import torch
import torchaudio
import os
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

def main():
    print("Starting simple audio generation script...")
    print(f"Using device: {device}")
    
    # Ensure the voices directory exists
    os.makedirs("voices", exist_ok=True)
    
    # Copy David Attenborough voice sample to voices directory if needed
    if not os.path.exists("voices/David_Attenborough.wav"):
        print("Copying David Attenborough voice sample to voices directory...")
        if os.path.exists("assets/David_Attenborough.wav"):
            import shutil
            shutil.copy("assets/David_Attenborough.wav", "voices/David_Attenborough.wav")
        else:
            print("Error: David_Attenborough.wav not found in assets directory")
            return
    
    # Load the model
    print("Loading Zonos model...")
    try:
        model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load the voice sample
    print("Loading David Attenborough voice sample...")
    try:
        wav, sampling_rate = torchaudio.load("voices/David_Attenborough.wav")
        print(f"Voice sample loaded: {wav.shape}, {sampling_rate}Hz")
    except Exception as e:
        print(f"Error loading voice sample: {e}")
        return
    
    # Generate speaker embedding
    print("Generating speaker embedding...")
    try:
        speaker = model.make_speaker_embedding(wav, sampling_rate)
        print("Speaker embedding generated successfully")
    except Exception as e:
        print(f"Error generating speaker embedding: {e}")
        return
    
    # Set a fixed seed for reproducibility
    torch.manual_seed(421)
    
    # Generate text
    text = "Hello, world!"
    print(f"Generating audio for text: '{text}'")
    
    # Create conditioning dictionary
    try:
        cond_dict = make_cond_dict(text=text, speaker=speaker, language="en-us")
        print("Conditioning dictionary created successfully")
    except Exception as e:
        print(f"Error creating conditioning dictionary: {e}")
        return
    
    # Prepare conditioning
    try:
        conditioning = model.prepare_conditioning(cond_dict)
        print("Conditioning prepared successfully")
    except Exception as e:
        print(f"Error preparing conditioning: {e}")
        return
    
    # Generate codes
    try:
        print("Generating audio codes...")
        codes = model.generate(conditioning)
        print("Audio codes generated successfully")
    except Exception as e:
        print(f"Error generating audio codes: {e}")
        return
    
    # Decode to audio
    try:
        print("Decoding audio...")
        wavs = model.autoencoder.decode(codes).cpu()
        print("Audio decoded successfully")
    except Exception as e:
        print(f"Error decoding audio: {e}")
        return
    
    # Save the audio
    output_file = "david_hello_world.wav"
    try:
        print(f"Saving audio to {output_file}...")
        torchaudio.save(output_file, wavs[0], model.autoencoder.sampling_rate)
        print(f"Audio saved successfully to {output_file}")
    except Exception as e:
        print(f"Error saving audio: {e}")
        return
    
    print("Audio generation completed successfully!")

if __name__ == "__main__":
    main()
