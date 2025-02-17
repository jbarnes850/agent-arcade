# Getting Started with Agent Arcade

This guide will help you get up and running with Agent Arcade, including troubleshooting common installation issues.

## System Requirements

Before you begin, ensure your system meets these requirements:

- **Python**: Version 3.8 - 3.12 (3.13 not yet supported)
- **Operating System**: Linux, macOS, or WSL2 on Windows
- **Node.js & npm**: Version 14 or higher (for NEAR CLI)
- **Storage**: At least 2GB free space
- **Memory**: At least 4GB RAM recommended

## Installation

1. **Clone the Repository**:
```bash
git clone https://github.com/your-username/agent-arcade.git
cd agent-arcade
```

2. **Run the Installation Script**:
```bash
./install.sh
```

The script will:
- Create a Python virtual environment
- Install all required dependencies
- Set up Atari ROMs
- Install and configure NEAR CLI

## Troubleshooting Installation

### Python Version Issues

If you see Python version errors:

1. **Check Current Version**:
```bash
python3 --version
```

2. **Install Compatible Version**:

On macOS:
```bash
brew install python@3.12
brew link python@3.12
```

On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install python3.12 python3.12-venv
```

On Windows (WSL2):
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.12 python3.12-venv
```

### Atari ROM Installation Issues

If you encounter ROM installation problems:

1. **Install Dependencies in Order**:
```bash
# First, install gymnasium with Atari support
pip install "gymnasium[atari]==0.28.1"

# Then install ALE-py
pip install "ale-py==0.8.1"

# Finally install AutoROM
pip install "AutoROM[accept-rom-license]==0.6.1"
```

2. **Install ROMs**:
```bash
# Method 1: Using AutoROM (preferred)
python3 -m AutoROM --accept-license

# Method 2: Using pip package
pip install autorom.accept-rom-license

# Method 3: Manual Installation
# Only use this if methods 1 and 2 fail
ROMS_DIR="$HOME/.local/lib/python3.*/site-packages/ale_py/roms"
mkdir -p "$ROMS_DIR"
wget https://github.com/openai/atari-py/raw/master/atari_py/atari_roms/pong.bin -P "$ROMS_DIR"
wget https://github.com/openai/atari-py/raw/master/atari_py/atari_roms/space_invaders.bin -P "$ROMS_DIR"
```

3. **Verify Installation**:

```bash
# Verify ALE interface
python3 -c "from ale_py import ALEInterface; ALEInterface()"

# Test specific games
python3 -c "import gymnasium; gymnasium.make('ALE/Pong-v5')"
python3 -c "import gymnasium; gymnasium.make('ALE/SpaceInvaders-v5')"
```

4. **Common ROM Issues**:
   - **ROM not found**: Make sure ROMs are in the correct directory
   - **Permission errors**: Check directory permissions with `ls -la $HOME/.local/lib/python3.*/site-packages/ale_py/roms`
   - **Import errors**: Ensure packages are installed in the correct order
   - **Version conflicts**: Use the exact versions specified above

### Package Installation Issues

1. **Clean Installation**:

```bash
# Remove existing virtual environment
rm -rf drl-env

# Create new environment
python3 -m venv drl-env
source drl-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install with verbose output
pip install -e . -v
```

2. **Dependency Conflicts**:

```bash
# Install specific versions
pip install "gymnasium[atari]==0.28.1"
pip install "stable-baselines3==2.0.0"
pip install "ale-py==0.8.1"
pip install "AutoROM[accept-rom-license]==0.6.1"
```

### NEAR CLI Issues

1. **Node.js Installation**:

On macOS:
```bash
brew install node@14
```

On Ubuntu/Debian:
```bash
curl -fsSL https://deb.nodesource.com/setup_14.x | sudo -E bash -
sudo apt-get install -y nodejs
```

2. **NEAR CLI Installation**:

```bash
# Remove existing installation
npm uninstall -g near-cli

# Clear npm cache
npm cache clean --force

# Install NEAR CLI
npm install -g near-cli
```

## First Steps

After successful installation:

1. **Verify CLI Installation**:
```bash
agent-arcade --version
```

2. **List Available Games**:
```bash
agent-arcade list-games
```

3. **Train Your First Agent**:
```bash
# Train Pong agent with visualization
agent-arcade train pong --render
```

4. **Evaluate Your Agent**:
```bash
agent-arcade evaluate pong --model models/pong_final.zip
```

5. **Login to NEAR Wallet**:
```bash
agent-arcade login
```

6. **Stake on Performance**:
```bash
agent-arcade stake pong --model models/pong_final.zip --amount 10 --target-score 15
```

## Common Error Messages

1. **"ImportError: No module named 'imp'"**:
   - This error occurs with Python 3.13
   - Solution: Use Python 3.12 or lower

2. **"ModuleNotFoundError: No module named 'ale_py'"**:
   - Solution: Reinstall ALE-py
   ```bash
   pip install ale-py==0.8.1
   ```

3. **"Error: Cannot find module 'near-api-js'"**:
   - Solution: Reinstall NEAR CLI
   ```bash
   npm install -g near-cli
   ```

4. **"ROM not found"**:
   - Solution: Follow manual ROM installation steps above

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/your-username/agent-arcade/issues)
2. Join our [Discord Community](https://discord.gg/your-invite)
3. Create a new issue with:
   - Your system information
   - Error message
   - Steps to reproduce
   - Logs from `install.sh`

## Quick Start

```bash
# Install Agent Arcade
git clone https://github.com/jbarnes850/agent-arcade.git
cd agent-arcade
./install.sh

# List available games
agent-arcade list-games

# Train your first agent (single-player)
agent-arcade train pong --render

# Train your first versus AI (multi-agent)
agent-arcade train pong-versus --train-paddle left --render
```

## Game Types

Agent Arcade supports two types of games:

1. **Single-Player Games**
   ```bash
   # Train a single AI agent
   agent-arcade train pong --render
   
   # Evaluate the trained agent
   agent-arcade evaluate pong --model models/pong_final.zip
   ```

2. **Versus Games (Multi-Agent)**
   ```bash
   # Train left paddle AI
   agent-arcade train pong-versus --train-paddle left --render
   
   # Train right paddle AI
   agent-arcade train pong-versus --train-paddle right --render
   
   # Have them compete
   agent-arcade compete pong-versus \
       --left-ai models/pong_left_final.zip \
       --right-ai models/pong_right_final.zip
   ```
