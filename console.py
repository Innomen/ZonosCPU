#!/usr/bin/env python3
"""
Console interface for Zonos CPU-only fork
This provides a text-based interface for voice cloning that doesn't require tkinter
"""

import os
import sys
import time
import shutil
import traceback
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Zonos modules
import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE

def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the application header"""
    print("=" * 60)
    print("             ZONOS VOICE CLONING (CPU-ONLY)              ")
    print("=" * 60)
    print()

def get_voice_files():
    """Get a list of available voice files"""
    voices_dir = Path(__file__).parent / "voices"
    voices_dir.mkdir(exist_ok=True)
    
    # Check for David Attenborough sample
    david_sample = Path(__file__).parent / "assets" / "David_Attenborough.wav"
    if david_sample.exists():
        david_dest = voices_dir / "David_Attenborough.wav"
        if not david_dest.exists():
            shutil.copy(str(david_sample), str(david_dest))
    
    voice_files = []
    for ext in ['.wav', '.mp3', '.flac', '.ogg']:
        voice_files.extend(list(voices_dir.glob(f"*{ext}")))
    
    return voice_files

def select_voice():
    """Let the user select a voice file"""
    voice_files = get_voice_files()
    
    if not voice_files:
        print("No voice files found in the 'voices' directory.")
        print("You'll need to provide a path to a voice file.")
        path = input("Enter path to voice file: ").strip()
        if os.path.exists(path):
            return path, None
        else:
            print(f"Error: File not found at {path}")
            return None, None
    
    print("Available voices:")
    for i, voice in enumerate(voice_files):
        print(f"{i+1}. {voice.stem}")
    
    print(f"{len(voice_files)+1}. Use a different file")
    
    try:
        choice = int(input("Select a voice (number): "))
        if 1 <= choice <= len(voice_files):
            return str(voice_files[choice-1]), voice_files[choice-1].stem
        elif choice == len(voice_files)+1:
            path = input("Enter path to voice file: ").strip()
            if os.path.exists(path):
                return path, None
            else:
                print(f"Error: File not found at {path}")
                return None, None
        else:
            print("Invalid choice.")
            return None, None
    except ValueError:
        print("Please enter a number.")
        return None, None

def get_text_input(voice_name=None):
    """Get text input from the user"""
    # Set default text for David Attenborough voice
    if voice_name and "david_attenborough" in voice_name.lower():
        default_text = "Hello world!"
        print(f"\nDefault text for {voice_name}: '{default_text}'")
        use_default = input("Use default text? (y/n, default: y): ").lower() != 'n'
        if use_default:
            return default_text
    
    print("\nEnter the text you want to synthesize (type 'END' on a new line when finished):")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    
    return "\n".join(lines)

def chunk_text(text, max_length=150):
    """Split text into chunks based on sentences"""
    # Split by sentence endings
    sentences = []
    current = ""
    
    # Simple sentence splitting
    for char in text:
        current += char
        if char in ['.', '!', '?'] and len(current.strip()) > 0:
            sentences.append(current)
            current = ""
    
    # Add any remaining text as a sentence
    if current.strip():
        sentences.append(current)
    
    # Combine sentences into chunks
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def process_audio(voice_path, text, model=None, chunk=True):
    """Process audio using the Zonos model"""
    try:
        print(f"Using voice: {os.path.basename(voice_path)}")
        print(f"Text length: {len(text)} characters")
        
        # Initialize model if not provided
        if model is None:
            print("Initializing Zonos model...")
            model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cpu")
        
        # Load the audio file
        print(f"Processing audio from: {voice_path}")
        print(f"Text to synthesize: {text}")
        wav, sampling_rate = torchaudio.load(voice_path)
        
        # Generate speaker embedding
        speaker = model.make_speaker_embedding(wav, sampling_rate)
        
        # Process text in chunks if needed
        if chunk and len(text) > 150:
            print("Chunking text into smaller segments...")
            chunks = chunk_text(text)
            all_wavs = []
            
            for i, chunk_text in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}: {chunk_text[:30]}...")
                
                # Create conditioning dictionary
                cond_dict = make_cond_dict(text=chunk_text, speaker=speaker, language="en-us")
                
                # Prepare conditioning
                conditioning = model.prepare_conditioning(cond_dict)
                
                # Generate codes
                print(f"Generating audio codes...")
                codes = model.generate(conditioning)
                
                # Decode codes to audio
                print(f"Decoding audio...")
                chunk_wav = model.autoencoder.decode(codes)
                all_wavs.append(chunk_wav)
            
            # Concatenate all chunks
            if all_wavs:
                print(f"Combining {len(all_wavs)} audio chunks...")
                wavs = torch.cat(all_wavs, dim=2)
            else:
                raise ValueError("No audio chunks were processed")
        else:
            # Create conditioning dictionary
            cond_dict = make_cond_dict(text=text, speaker=speaker, language="en-us")
            
            # Prepare conditioning
            conditioning = model.prepare_conditioning(cond_dict)
            
            # Generate codes
            codes = model.generate(conditioning)
            
            # Decode codes to audio
            wavs = model.autoencoder.decode(codes)
        
        # Save the audio file
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.wav")
        if wavs.dim() == 3:
            wavs = wavs[0]  # Take the first batch item if batched
        torchaudio.save(output_path, wavs.cpu(), model.autoencoder.sampling_rate)
        
        return output_path
    except Exception as e:
        print(f"ERROR: Failed to process audio: {e}")
        traceback.print_exc()
        return None

def main():
    """Main function for the console interface"""
    clear_screen()
    print_header()
    
    print("Welcome to the Zonos Voice Cloning console interface!")
    print("This application allows you to clone voices using CPU-only processing.")
    print()
    
    # Select voice
    voice_path = None
    voice_name = None
    while not voice_path:
        voice_path, voice_name = select_voice()
        if not voice_path:
            if input("Try again? (y/n): ").lower() != 'y':
                print("Exiting.")
                return
    
    print(f"\nUsing voice: {os.path.basename(voice_path)}")
    
    # Get text input
    text = get_text_input(voice_name)
    if not text:
        text = "Hello world!"
        print(f"\nUsing default text: '{text}'")
    
    # Ask about chunking
    chunk = input("\nAutomatically chunk text into sentences? (y/n, default: y): ").lower() != 'n'
    
    # Process audio
    print("\nProcessing audio... This may take a while.")
    output_path = process_audio(voice_path, text, chunk=chunk)
    
    if output_path:
        abs_path = os.path.abspath(output_path)
        print(f"\nAudio saved to: {abs_path}")
        
        # Ask if user wants to save to a different location
        if input("\nSave to a different location? (y/n): ").lower() == 'y':
            new_path = input("Enter new path (including filename): ").strip()
            try:
                shutil.copy(output_path, new_path)
                print(f"Audio saved to: {new_path}")
            except Exception as e:
                print(f"Error saving file: {e}")
    
    print("\nThank you for using Zonos Voice Cloning!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
