#!/usr/bin/env fish
# Unified launcher script for Zonos CPU-only fork (Fish Shell Version)
# This script is specifically designed for fish shell users
# English-only version with simplified dependencies

# Function to check if virtual environment is active
function is_venv_active
    if test -n "$VIRTUAL_ENV"
        return 0  # True
    else
        return 1  # False
    end
end

# Function to check if a command exists
function command_exists
    command -v $argv[1] >/dev/null 2>&1
end

# Function to create and activate virtual environment
function setup_venv
    set venv_dir "venv"
    
    # Create virtual environment if it doesn't exist
    if not test -d "$venv_dir"
        echo "Creating virtual environment..."
        python -m venv "$venv_dir"
    end
    
    # Activate virtual environment
    echo "Activating virtual environment..."
    source "$venv_dir/bin/activate.fish" || \
    echo "Failed to activate virtual environment. Please run: source $venv_dir/bin/activate.fish"
    
    # Verify activation
    if test -z "$VIRTUAL_ENV"
        echo "Warning: Virtual environment not activated. Using system Python."
    else
        echo "Successfully activated virtual environment: $VIRTUAL_ENV"
    end
end

# Function to install dependencies
function install_dependencies
    echo "Installing dependencies..."
    
    # Make sure we're in the virtual environment
    if test -z "$VIRTUAL_ENV"
        echo "Error: Not in a virtual environment. Dependencies may not install correctly."
        return
    end
    
    # Install required packages one by one to better handle errors
    # Simplified English-only dependencies
    pip install torch || echo "Warning: Failed to install torch"
    pip install torchaudio || echo "Warning: Failed to install torchaudio"
    pip install transformers || echo "Warning: Failed to install transformers"
    pip install safetensors || echo "Warning: Failed to install safetensors"
    pip install huggingface_hub || echo "Warning: Failed to install huggingface_hub"
    pip install tqdm || echo "Warning: Failed to install tqdm"
    pip install inflect || echo "Warning: Failed to install inflect"
    
    # Check for tkinter
    python -c "import tkinter" 2>/dev/null
    if test $status -ne 0
        echo "Warning: tkinter is not installed. GUI interface will not work."
        echo "To install tkinter on Garuda Linux: sudo pacman -S tk"
        echo "Continuing with console interface..."
    end
end

# Function to check for GPU availability
function check_gpu
    if command_exists nvidia-smi
        echo "GPU detected, but this is a CPU-only fork. Will use CPU for processing."
    else
        echo "No GPU detected. Using CPU for processing."
    end
end

# Function to create necessary directories
function setup_directories
    # Create voices directory if it doesn't exist
    mkdir -p voices
    
    # Copy David Attenborough sample if available
    if test -f "assets/David_Attenborough.wav"
        if not test -f "voices/David_Attenborough.wav"
            echo "Copying David Attenborough voice sample to voices directory..."
            cp "assets/David_Attenborough.wav" "voices/David_Attenborough.wav"
        end
    end
end

# Function to run the application
function run_app
    echo "Starting Zonos CPU-only fork (English-only version)..."
    
    # Check if tkinter is available
    python -c "import tkinter" 2>/dev/null
    if test $status -eq 0
        echo "Using GUI interface"
        python gui.py
    else
        echo "Using console interface"
        python console.py
    end
end

# Main execution
function main
    echo "Detected shell: fish"
    echo "Running English-only version with simplified dependencies"
    
    # Check if virtual environment is active
    if not is_venv_active
        setup_venv
    else
        echo "Virtual environment already active: $VIRTUAL_ENV"
    end
    
    # Install dependencies
    install_dependencies
    
    # Check for GPU
    check_gpu
    
    # Setup directories
    setup_directories
    
    # Run the application
    run_app
end

# Execute main function
main
