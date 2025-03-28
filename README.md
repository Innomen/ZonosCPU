# Zonos CPU-Only Fork

This is a CPU-only fork of the [Zonos voice cloning application](https://github.com/Zyphra/Zonos) that provides a user-friendly interface for voice cloning without requiring a GPU.

## Features

- Works on CPU-only systems without requiring a GPU
- Provides both GUI and console interfaces
- Automatically detects and adapts to different shell environments (bash, zsh, fish)
- Includes automatic virtual environment detection and setup
- Supports text chunking for processing longer texts
- Includes the David Attenborough voice sample for easy testing

## Installation

1. Clone this repository:
```
git clone https://github.com/Innomen/zonos-cpu-fork.git
cd zonos-cpu-fork
```

2. Run the launcher script:
```
bash run.sh
```

The launcher script will:
- Detect your shell environment
- Create and activate a virtual environment if needed
- Install required dependencies
- Check for tkinter availability
- Launch the appropriate interface (GUI or console)

## Requirements

- Python 3.8 or higher
- For GUI interface: tkinter (installation instructions provided by the launcher script)
- Internet connection for downloading the Zonos model (first run only)

## Usage

### GUI Interface

The GUI interface provides a user-friendly way to interact with the voice cloning functionality:

1. Select a voice from the dropdown or upload a new voice file
2. Enter the text you want to synthesize
3. Choose whether to automatically chunk text into sentences
4. Click "Generate Audio" to create the audio
5. Use "Play Audio" to listen to the generated audio
6. Use "Save Audio" to save the audio to a specific location

### Console Interface

The console interface provides the same functionality in a text-based environment:

1. Select a voice from the list or provide a path to a voice file
2. Enter the text you want to synthesize
3. Choose whether to automatically chunk text into sentences
4. The audio will be generated and saved to the output.wav file
5. You can choose to save the audio to a different location

## Voice Samples

Voice samples are stored in the `voices` directory. The David Attenborough voice sample is included by default.

To add your own voice samples:
1. Place WAV, MP3, FLAC, or OGG files in the `voices` directory
2. Refresh the voice list in the GUI or restart the application

## Troubleshooting

If you encounter any issues:

1. Make sure you have the required dependencies installed
2. Check that you have sufficient disk space for the Zonos model (~3GB)
3. For GUI issues, ensure tkinter is properly installed
4. Check the log output for specific error messages

## Credits

This is a fork of the [Zonos voice cloning application](https://github.com/Zyphra/Zonos) by Zyphra, modified to work better on CPU-only systems.
