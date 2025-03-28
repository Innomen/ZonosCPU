#!/usr/bin/env python3
"""
GUI interface for Zonos CPU-only fork
This provides a user-friendly interface for voice cloning
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import threading
import subprocess
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

class ZonosGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Zonos Voice Cloning (CPU-only)")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
        # Set up variables
        self.audio_path = tk.StringVar()
        self.text_to_speak = tk.StringVar(value="Hello world!")
        self.status = tk.StringVar(value="Ready")
        self.processing = False
        self.output_path = None
        self.selected_voice_name = None
        self.model = None
        self.last_error = None
        
        # Create voice samples directory if it doesn't exist
        self.voices_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
        os.makedirs(self.voices_dir, exist_ok=True)
        
        # Copy David Attenborough sample if available
        david_sample = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "David_Attenborough.wav")
        if os.path.exists(david_sample):
            david_dest = os.path.join(self.voices_dir, "David_Attenborough.wav")
            if not os.path.exists(david_dest):
                shutil.copy(david_sample, david_dest)
                print("Copying David Attenborough voice sample to voices directory...")
        
        # Create the GUI
        self.create_widgets()
        
        # Populate voice dropdown
        self.refresh_voices()
        
    def create_widgets(self):
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Zonos Voice Cloning (CPU-only)", font=("Arial", 18, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Voice selection section
        voice_frame = ttk.LabelFrame(main_frame, text="Voice Selection", padding="10")
        voice_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Voice dropdown
        voice_label = ttk.Label(voice_frame, text="Select Voice:")
        voice_label.grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.voice_dropdown = ttk.Combobox(voice_frame, width=40)
        self.voice_dropdown.grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)
        self.voice_dropdown.bind("<<ComboboxSelected>>", self.on_voice_selected)
        
        refresh_button = ttk.Button(voice_frame, text="Refresh", command=self.refresh_voices)
        refresh_button.grid(row=0, column=2, sticky=tk.W, pady=5)
        
        # Upload new voice
        upload_label = ttk.Label(voice_frame, text="Upload New Voice:")
        upload_label.grid(row=1, column=0, sticky=tk.W, pady=5)
        
        upload_entry = ttk.Entry(voice_frame, textvariable=self.audio_path, width=40)
        upload_entry.grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)
        
        browse_button = ttk.Button(voice_frame, text="Browse", command=self.browse_audio)
        browse_button.grid(row=1, column=2, sticky=tk.W, pady=5)
        
        # Text input section
        text_frame = ttk.LabelFrame(main_frame, text="Text to Speak", padding="10")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.text_input = tk.Text(text_frame, wrap=tk.WORD, height=10)
        self.text_input.pack(fill=tk.BOTH, expand=True, pady=5)
        self.text_input.insert(tk.END, "Hello world!")
        
        # Options section
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Chunk text option
        self.chunk_text = tk.BooleanVar(value=True)
        chunk_check = ttk.Checkbutton(options_frame, text="Automatically chunk text into sentences", variable=self.chunk_text)
        chunk_check.grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        generate_button = ttk.Button(button_frame, text="Generate Audio", command=self.generate_audio)
        generate_button.pack(side=tk.LEFT, padx=5)
        
        play_button = ttk.Button(button_frame, text="Play Audio", command=self.play_audio)
        play_button.pack(side=tk.LEFT, padx=5)
        
        save_button = ttk.Button(button_frame, text="Save Audio", command=self.save_audio)
        save_button.pack(side=tk.LEFT, padx=5)
        
        # Log section
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Add a frame for the log text and buttons
        log_content_frame = ttk.Frame(log_frame)
        log_content_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_content_frame, wrap=tk.WORD, height=5)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=5)
        self.log_text.config(state=tk.DISABLED)
        
        # Add scrollbar for log text
        log_scrollbar = ttk.Scrollbar(log_content_frame, command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        
        # Add copy button for log text
        log_button_frame = ttk.Frame(log_frame)
        log_button_frame.pack(fill=tk.X)
        
        self.copy_button = ttk.Button(log_button_frame, text="Copy to Clipboard", command=self.copy_log_to_clipboard)
        self.copy_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X)
        
        status_label = ttk.Label(status_frame, textvariable=self.status)
        status_label.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode="indeterminate")
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
    
    def copy_log_to_clipboard(self):
        """Copy log content to clipboard"""
        log_content = self.log_text.get("1.0", tk.END).strip()
        if log_content:
            self.root.clipboard_clear()
            self.root.clipboard_append(log_content)
            self.status.set("Log copied to clipboard")
            
            # If there was an error, also copy the error specifically
            if self.last_error:
                self.root.clipboard_clear()
                self.root.clipboard_append(self.last_error)
                self.status.set("Error message copied to clipboard")
    
    def log(self, message):
        """Add a message to the log"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update_idletasks()
    
    def refresh_voices(self):
        """Refresh the list of available voices"""
        voices = []
        voice_paths = {}
        if os.path.exists(self.voices_dir):
            for file in os.listdir(self.voices_dir):
                if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    voice_name = os.path.splitext(file)[0]
                    voices.append(voice_name)
                    voice_paths[voice_name] = os.path.join(self.voices_dir, file)
        
        self.voice_dropdown['values'] = voices
        self.voice_paths = voice_paths
        if voices:
            self.voice_dropdown.current(0)
            self.on_voice_selected(None)
    
    def on_voice_selected(self, event):
        """Handle voice selection"""
        selected_voice = self.voice_dropdown.get()
        self.selected_voice_name = selected_voice
        
        # Set default text for David Attenborough voice
        if selected_voice and "david_attenborough" in selected_voice.lower():
            self.text_input.delete("1.0", tk.END)
            self.text_input.insert(tk.END, "Hello world!")
            self.log(f"Set default text for {selected_voice}")
    
    def browse_audio(self):
        """Open file dialog to select audio file"""
        filetypes = (
            ('Audio files', '*.wav *.mp3 *.flac *.ogg'),
            ('All files', '*.*')
        )
        
        filename = filedialog.askopenfilename(
            title='Select a voice sample',
            initialdir='/',
            filetypes=filetypes
        )
        
        if filename:
            self.audio_path.set(filename)
            
            # Ask if user wants to save this voice for future use
            if messagebox.askyesno("Save Voice", "Would you like to save this voice for future use?"):
                voice_name = os.path.basename(filename)
                voice_name = os.path.splitext(voice_name)[0]
                
                # Ask for a name for the voice
                new_name = simpledialog.askstring("Voice Name", "Enter a name for this voice:", initialvalue=voice_name)
                if new_name:
                    # Copy the file to the voices directory
                    dest_path = os.path.join(self.voices_dir, f"{new_name}{os.path.splitext(filename)[1]}")
                    shutil.copy(filename, dest_path)
                    self.refresh_voices()
                    self.voice_dropdown.set(new_name)
                    self.selected_voice_name = new_name
    
    def initialize_model(self):
        """Initialize the Zonos model if not already initialized"""
        if self.model is None:
            try:
                self.log("Initializing Zonos model...")
                self.model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cpu")
                self.log("Model initialized successfully")
                return True
            except Exception as e:
                error_message = f"Error initializing model: {e}"
                self.log(error_message)
                
                # Store the full error message for clipboard copying
                self.last_error = error_message + "\n" + traceback.format_exc()
                self.log(traceback.format_exc())
                
                # Show a message box with copy button suggestion
                messagebox.showerror("Model Initialization Error", 
                                    f"{error_message}\n\nClick 'Copy to Clipboard' button to copy the full error message.")
                return False
        return True
    
    def generate_audio(self):
        """Generate audio from text using selected voice"""
        if self.processing:
            messagebox.showwarning("Processing", "Already processing audio. Please wait.")
            return
        
        # Get the text from the text input
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Input Error", "Please enter text to speak.")
            return
        
        # Get the voice file
        voice_file = None
        selected_voice = self.voice_dropdown.get()
        
        if selected_voice and selected_voice in self.voice_paths:
            voice_file = self.voice_paths[selected_voice]
        
        # If no voice is selected, use the uploaded file
        if not voice_file and self.audio_path.get():
            voice_file = self.audio_path.get()
        
        if not voice_file:
            messagebox.showwarning("Input Error", "Please select or upload a voice sample.")
            return
        
        # Start processing in a separate thread
        self.processing = True
        self.status.set("Processing audio...")
        self.progress.start()
        self.log(f"Starting voice cloning with voice: {os.path.basename(voice_file)}")
        self.log(f"Text length: {len(text)} characters")
        
        # Clear any previous error
        self.last_error = None
        
        # Start processing thread
        threading.Thread(target=self._process_audio, args=(text, voice_file), daemon=True).start()
    
    def _process_audio(self, text, voice_file):
        """Process audio in a separate thread"""
        try:
            # Initialize model if needed
            if not self.initialize_model():
                self.processing = False
                self.status.set("Error initializing model")
                self.progress.stop()
                return
            
            # Load voice sample
            wav, sr = torchaudio.load(voice_file)
            speaker = self.model.make_speaker_embedding(wav, sr)
            
            # Create conditioning dictionary
            cond_dict = make_cond_dict(text=text, speaker=speaker, language="en-us")
            conditioning = self.model.prepare_conditioning(cond_dict)
            
            # Generate audio
            self.log("Generating audio...")
            codes = self.model.generate(conditioning)
            
            # Decode audio
            self.log("Decoding audio...")
            wavs = self.model.autoencoder.decode(codes).cpu()
            
            # Save temporary output
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
            os.makedirs(output_dir, exist_ok=True)
            
            self.output_path = os.path.join(output_dir, "output.wav")
            torchaudio.save(self.output_path, wavs[0], self.model.autoencoder.sampling_rate)
            
            self.log(f"Audio generated successfully: {self.output_path}")
            self.status.set("Audio generation complete")
            
        except Exception as e:
            error_message = f"Error generating audio: {e}"
            self.log(error_message)
            
            # Store the full error message for clipboard copying
            self.last_error = error_message + "\n" + traceback.format_exc()
            self.log(traceback.format_exc())
            
            # Show a message box with copy button suggestion
            self.root.after(0, lambda: messagebox.showerror("Audio Generation Error", 
                                    f"{error_message}\n\nClick 'Copy to Clipboard' button to copy the full error message."))
            self.status.set("Error generating audio")
        
        finally:
            self.processing = False
            self.progress.stop()
    
    def play_audio(self):
        """Play the generated audio"""
        if not self.output_path or not os.path.exists(self.output_path):
            messagebox.showwarning("Playback Error", "No audio has been generated yet.")
            return
        
        # Use system default audio player
        try:
            if sys.platform == "win32":
                os.startfile(self.output_path)
            elif sys.platform == "darwin":  # macOS
                subprocess.call(["open", self.output_path])
            else:  # Linux
                subprocess.call(["xdg-open", self.output_path])
        except Exception as e:
            messagebox.showerror("Playback Error", f"Error playing audio: {e}")
    
    def save_audio(self):
        """Save the generated audio to a file"""
        if not self.output_path or not os.path.exists(self.output_path):
            messagebox.showwarning("Save Error", "No audio has been generated yet.")
            return
        
        # Open file dialog to select save location
        filetypes = (
            ('WAV files', '*.wav'),
            ('All files', '*.*')
        )
        
        filename = filedialog.asksaveasfilename(
            title='Save audio as',
            initialdir='/',
            defaultextension='.wav',
            filetypes=filetypes
        )
        
        if filename:
            try:
                shutil.copy(self.output_path, filename)
                self.log(f"Audio saved to: {filename}")
                messagebox.showinfo("Save Complete", f"Audio saved to: {filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Error saving audio: {e}")

def main():
    # Check if GPU is available but use CPU anyway (this is a CPU-only fork)
    if torch.cuda.is_available():
        print("GPU detected, but this is a CPU-only fork. Will use CPU for processing.")
    
    # Create the main window
    root = tk.Tk()
    app = ZonosGUI(root)
    
    # Start the main loop
    print("Starting Zonos CPU-only fork (English-only version)...")
    print("Using GUI interface")
    root.mainloop()

if __name__ == "__main__":
    main()
