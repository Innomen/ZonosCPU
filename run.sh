#!/bin/bash
# Unified launcher script for Zonos CPU-only fork
# This script can detect and adapt to different shell environments
# English-only version with simplified dependencies

# Function to detect shell type
detect_shell() {
    if [ -n "$FISH_VERSION" ]; then
        echo "fish"
    elif [ -n "$BASH_VERSION" ]; then
        echo "bash"
    elif [ -n "$ZSH_VERSION" ]; then
        echo "zsh"
    else
        echo "unknown"
    fi
}

# Function to check if virtual environment is active
is_venv_active() {
    if [ -n "$VIRTUAL_ENV" ]; then
        return 0  # True
    else
        return 1  # False
    fi
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to create and activate virtual environment based on shell type
setup_venv() {
    local shell_type=$1
    local venv_dir="venv"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$venv_dir" ]; then
        echo "Creating virtual environment..."
        python3 -m venv "$venv_dir"
    fi
    
    # Activate virtual environment based on shell type
    echo "Activating virtual environment..."
    case "$shell_type" in
        "fish")
            # For fish shell
            source "$venv_dir/bin/activate.fish" || \
            echo "Failed to activate virtual environment for fish shell. Please run: source $venv_dir/bin/activate.fish"
            ;;
        "bash"|"zsh"|*)
            # For bash, zsh, or unknown shells
            source "$venv_dir/bin/activate" || \
            echo "Failed to activate virtual environment. Please run: source $venv_dir/bin/activate"
            ;;
    esac
    
    # Verify activation
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "Warning: Virtual environment not activated. Using system Python."
    else
        echo "Successfully activated virtual environment: $VIRTUAL_ENV"
    fi
}

# Function to install dependencies
install_dependencies() {
    echo "Installing dependencies..."
    
    # Make sure we're in the virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "Error: Not in a virtual environment. Dependencies may not install correctly."
        return
    fi
    
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
    if [ $? -ne 0 ]; then
        echo "Warning: tkinter is not installed. GUI interface will not work."
        echo "To install tkinter:"
        
        # Detect OS and provide appropriate installation instructions
        if [ -f /etc/debian_version ]; then
            echo "  For Debian/Ubuntu: sudo apt-get install python3-tk"
        elif [ -f /etc/redhat-release ]; then
            echo "  For Fedora/RHEL: sudo dnf install python3-tkinter"
        elif [ -f /etc/arch-release ] || command_exists pacman; then
            echo "  For Arch/Garuda: sudo pacman -S tk"
        elif [ -f /etc/SuSE-release ]; then
            echo "  For openSUSE: sudo zypper install python3-tk"
        elif command_exists brew; then
            echo "  For macOS with Homebrew: brew install python-tk"
        elif [ "$(uname)" = "Darwin" ]; then
            echo "  For macOS: Install Python with tkinter support from python.org"
        else
            echo "  Please install tkinter for your operating system"
        fi
        
        echo "Continuing with console interface..."
    fi
}

# Function to check for GPU availability
check_gpu() {
    if command_exists nvidia-smi; then
        echo "GPU detected, but this is a CPU-only fork. Will use CPU for processing."
    else
        echo "No GPU detected. Using CPU for processing."
    fi
}

# Function to create necessary directories
setup_directories() {
    # Create voices directory if it doesn't exist
    mkdir -p voices
    
    # Copy David Attenborough sample if available
    if [ -f "assets/David_Attenborough.wav" ]; then
        if [ ! -f "voices/David_Attenborough.wav" ]; then
            echo "Copying David Attenborough voice sample to voices directory..."
            cp "assets/David_Attenborough.wav" "voices/David_Attenborough.wav"
        fi
    fi
}

# Function to run the application
run_app() {
    echo "Starting Zonos CPU-only fork (English-only version)..."
    
    # Check if tkinter is available
    python -c "import tkinter" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "Using GUI interface"
        python gui.py
    else
        echo "Using console interface"
        python console.py
    fi
}

# Main execution
main() {
    local shell_type=$(detect_shell)
    echo "Detected shell: $shell_type"
    echo "Running English-only version with simplified dependencies"
    
    # Check if virtual environment is active
    if ! is_venv_active; then
        setup_venv "$shell_type"
    else
        echo "Virtual environment already active: $VIRTUAL_ENV"
    fi
    
    # Install dependencies
    install_dependencies
    
    # Check for GPU
    check_gpu
    
    # Setup directories
    setup_directories
    
    # Run the application
    run_app
}

# Execute main function
main
