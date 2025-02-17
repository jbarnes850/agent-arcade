#!/bin/bash
set -e

echo "🎮 Installing Agent Arcade..."

# Function to handle errors
handle_error() {
    echo "❌ Error occurred in install.sh:"
    echo "  Line: $1"
    echo "  Exit code: $2"
    echo "Please check the error message above and try again."
    exit 1
}

# Set up error handling
trap 'handle_error ${LINENO} $?' ERR

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check Python version is >= 3.9 and < 3.13
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [ "$(printf '%s\n' "3.9" "$python_version" | sort -V | head -n1)" != "3.9" ] || [ "$(printf '%s\n' "3.13" "$python_version" | sort -V | head -n1)" != "$python_version" ]; then
    echo "❌ Python version must be between 3.9 and 3.12. Found version: $python_version"
    exit 1
fi

# Check disk space before starting
echo "🔍 Checking system requirements..."
required_space=2048  # 2GB in MB
available_space=$(df -m . | awk 'NR==2 {print $4}')
if [ "$available_space" -lt "$required_space" ]; then
    echo "❌ Insufficient disk space. Required: 2GB, Available: $((available_space/1024))GB"
    exit 1
fi

# Check memory
total_memory=$(sysctl -n hw.memsize 2>/dev/null || free -b | awk '/^Mem:/{print $2}')
total_memory_gb=$((total_memory/1024/1024/1024))
if [ "$total_memory_gb" -lt 4 ]; then
    echo "⚠️  Warning: Less than 4GB RAM detected. Training performance may be impacted."
fi

# Check if virtual environment exists
if [ ! -d "drl-env" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv drl-env
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source drl-env/bin/activate || {
    echo "❌ Failed to activate virtual environment."
    exit 1
}

# Verify pip installation
echo "🔍 Verifying pip installation..."
if ! command -v pip &> /dev/null; then
    echo "❌ pip not found. Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py --user
    rm get-pip.py
fi

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip

# Clean install - remove existing packages if present
echo "🧹 Cleaning existing installations..."
pip uninstall -y agent-arcade ale-py shimmy gymnasium pettingzoo supersuit || true

# Install dependencies in correct order with error handling
echo "📥 Installing core dependencies..."

# Install PyTorch first
echo "Installing PyTorch..."
if ! pip install "torch>=2.3.0"; then
    echo "❌ Failed to install PyTorch."
    exit 1
fi

# Install Gymnasium with Atari support first
echo "Installing Gymnasium with Atari support..."
if ! pip install "gymnasium[atari]>=0.29.1" "gymnasium[accept-rom-license]>=0.29.1"; then
    echo "❌ Failed to install Gymnasium with Atari support."
    exit 1
fi

# Install ALE interface
echo "Installing latest ALE-py..."
if ! pip install "ale-py>=0.10.2"; then
    echo "❌ Failed to install ALE interface."
    exit 1
fi

# Install Shimmy for environment compatibility
echo "Installing Shimmy..."
if ! pip install "shimmy[atari]>=2.0.0"; then
    echo "❌ Failed to install Shimmy."
    exit 1
fi

# Install PettingZoo and SuperSuit
echo "Installing PettingZoo and SuperSuit..."
if ! pip install "pettingzoo[atari]>=1.24.1" "supersuit>=3.9.0"; then
    echo "❌ Failed to install PettingZoo and SuperSuit."
    exit 1
fi

# Install Stable-Baselines3 after environment dependencies
echo "Installing Stable-Baselines3..."
if ! pip install "stable-baselines3[extra]>=2.5.0"; then
    echo "❌ Failed to install Stable-Baselines3."
    exit 1
fi

# Install AutoROM for ROM management
echo "🎲 Installing AutoROM..."
if ! pip install "autorom>=0.6.1"; then
    echo "❌ Failed to install AutoROM."
    exit 1
fi

# Install Atari ROMs using AutoROM
echo "🎲 Installing Atari ROMs..."
if ! AutoROM --accept-license; then
    echo "❌ Failed to install ROMs using AutoROM."
    exit 1
fi

# Verify environment setup
echo "🎮 Verifying environment setup..."
python3 -c "
import gymnasium as gym
import ale_py
import pettingzoo
import supersuit as ss
from pettingzoo.atari import pong_v3

# Test PettingZoo environment
env = pong_v3.env(render_mode='rgb_array')
env = ss.color_reduction_v0(env)
env = ss.resize_v1(env, 84, 84)
env = ss.frame_stack_v1(env, 4)
print('✅ PettingZoo environment verified')

# Test ALE environment
gym.register_envs(ale_py)
print(f'A.L.E: Arcade Learning Environment (version {ale_py.__version__})')
print('✅ Environment registration successful')
" || {
    echo "❌ Environment verification failed."
    exit 1
}

# Install the agent-arcade package
echo "📥 Installing Agent Arcade..."
if ! pip install -e .; then
    echo "❌ Failed to install Agent Arcade package."
    exit 1
fi

# Ask if user wants to install NEAR integration
echo ""
echo "🌐 Would you like to install NEAR integration for staking? (y/N)"
read -r install_near

if [[ $install_near =~ ^[Yy]$ ]]; then
    echo "Installing NEAR integration..."
    if ! pip install -e ".[staking]"; then
        echo "❌ Failed to install NEAR integration."
        exit 1
    fi

    # Setup NEAR CLI if not installed
    if ! command -v near &> /dev/null; then
        if ! command -v npm &> /dev/null; then
            echo "❌ npm is required for NEAR CLI but not installed."
            echo "Please install Node.js and npm first: https://nodejs.org/"
            exit 1
        fi
        echo "🌐 Installing NEAR CLI..."
        npm install -g near-cli || {
            echo "❌ NEAR CLI installation failed."
            exit 1
        }
    fi
fi

# Create necessary directories
mkdir -p models tensorboard videos

# Verify Agent Arcade CLI installation
echo "✅ Verifying Agent Arcade CLI..."
agent-arcade --version || {
    echo "❌ Agent Arcade CLI verification failed."
    exit 1
}

echo "🎉 Installation complete! Get started with: agent-arcade --help"
echo ""
echo "📚 Available commands:"
echo "  agent-arcade train         - Train an agent for a game"
echo "  agent-arcade evaluate      - Evaluate a trained model"
echo "  agent-arcade leaderboard   - View game leaderboards"
if [[ $install_near =~ ^[Yy]$ ]]; then
    echo "  agent-arcade stake         - Manage stakes and evaluations"
    echo "  agent-arcade wallet-cmd    - Manage NEAR wallet"
fi
echo ""
echo "🎮 Try training your first agent:"
echo "  agent-arcade train pong-2p --role first_0 --render"
echo ""
echo "📊 Monitor training progress:"
echo "  tensorboard --logdir ./tensorboard"

# Print system information
echo "📊 System Information:"
echo "  - OS: $(uname -s) $(uname -r)"
echo "  - Python: $(python3 --version)"
echo "  - Pip: $(pip --version)"
echo "  - CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || cat /proc/cpuinfo | grep 'model name' | head -n1 | cut -d':' -f2)"
echo "  - Memory: ${total_memory_gb}GB"
echo "  - Disk Space: $((available_space/1024))GB available"

# Print installation summary
echo "📝 Installation Summary:"
echo "  - Virtual Environment: drl-env"
echo "  - ROM Directory: $(python3 -c "import ale_py; from pathlib import Path; print(Path(ale_py.__file__).parent / 'roms')")"
echo "  - Models Directory: $(pwd)/models"
echo "  - Tensorboard Logs: $(pwd)/tensorboard"
echo "  - Video Recordings: $(pwd)/videos" 