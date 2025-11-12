# Installing Python 3.11 for Alt ML Pipeline 1

The pipeline is configured to use Python 3.11 for optimal compatibility with ML libraries like PyTorch, Ultralytics, and MLflow.

## Why Python 3.11?

- **Best ML library support**: All major ML frameworks have stable wheels for Python 3.11
- **Performance**: Python 3.11 is ~25% faster than 3.10
- **Stability**: Mature version with excellent package compatibility
- **Not bleeding edge**: Python 3.13 is too new; some packages may not have wheels yet

## Check Current Python Version

```bash
python3 --version
python3.11 --version  # Check if 3.11 is already installed
```

## Installation Methods

### Option 1: Homebrew (macOS) - RECOMMENDED

```bash
# Install Python 3.11
brew install python@3.11

# Verify installation
python3.11 --version
# Should output: Python 3.11.x

# Link it (optional, makes it available as python3.11)
brew link python@3.11
```

### Option 2: pyenv - RECOMMENDED for Multiple Python Versions

This is the best option if you need to manage multiple Python versions on your system.

```bash
# Install pyenv (if not already installed)
# macOS:
brew install pyenv

# Linux:
curl https://pyenv.run | bash

# Add to shell profile (~/.bashrc, ~/.zshrc, etc.)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc

# Install Python 3.11.7 (latest 3.11 version)
pyenv install 3.11.7

# Set as local version for this project
cd /path/to/brain_buddies_mlm-25/alt_ml_pipeline_1
pyenv local 3.11.7

# Verify
python --version  # Should show 3.11.7
```

### Option 3: Official Python.org Installer (macOS/Windows)

1. Go to https://www.python.org/downloads/
2. Download Python 3.11.7 (or latest 3.11.x)
3. Run installer
4. Verify: `python3.11 --version`

### Option 4: apt (Ubuntu/Debian Linux)

```bash
# Add deadsnakes PPA (for latest Python versions)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-dev

# Verify
python3.11 --version
```

## After Installing Python 3.11

Once Python 3.11 is installed, run the setup script:

```bash
cd alt_ml_pipeline_1
bash scripts/setup_environment.sh
```

The script will:
1. Automatically detect Python 3.11
2. Create a virtual environment with Python 3.11
3. Install all required packages

## Troubleshooting

### "python3.11: command not found"

Check where Python 3.11 was installed:

```bash
# macOS/Linux
which python3.11
ls /usr/local/bin/python3.11
ls /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11

# If installed via Homebrew
ls $(brew --prefix)/bin/python3.11
```

If found but not in PATH, add to your shell profile:

```bash
# Example for Homebrew installation
echo 'export PATH="/usr/local/opt/python@3.11/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### "No module named 'venv'"

On some Linux systems, you need to install the venv module:

```bash
sudo apt install python3.11-venv
```

### Multiple Python Versions Conflict

Use pyenv (Option 2 above) to manage multiple versions cleanly. It allows you to:
- Have Python 3.11 for this project
- Keep Python 3.13 for other projects
- Switch between versions easily

```bash
# Set Python 3.11 only for this directory
cd alt_ml_pipeline_1
pyenv local 3.11.7

# Verify
python --version  # Shows 3.11.7 in this directory
cd ..
python --version  # Shows system default elsewhere
```

### Still Having Issues?

**Manual virtual environment creation**:

```bash
# If setup script fails, create venv manually
/path/to/python3.11 -m venv ~/venvs/alt_pipeline_1
source ~/venvs/alt_pipeline_1/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Verification

After setup, verify everything works:

```bash
source ~/venvs/alt_pipeline_1/bin/activate
python scripts/test_installation.py
```

Expected output:
```
========================================
Testing Package Imports
========================================

✓ PyTorch                 2.1.2
✓ TorchVision            0.16.2
✓ Ultralytics (YOLO)     8.0.227
✓ OpenCV                 4.8.1.78
...
```

## Why Not Python 3.13?

Python 3.13 (your current version) is very new (released October 2024):

**Potential Issues**:
- Some packages don't have prebuilt wheels yet
- May need to compile from source (slow, requires dev tools)
- Less tested in ML community
- Ultralytics YOLO may not have official support yet

**If you want to try Python 3.13 anyway**:
1. Skip this guide
2. Run `bash scripts/setup_environment.sh` with Python 3.13
3. If packages fail to install, fall back to Python 3.11

## Quick Reference

```bash
# Check installed Python versions
ls /usr/local/bin/python*
ls $(brew --prefix)/bin/python*

# Install Python 3.11 (macOS)
brew install python@3.11

# Install Python 3.11 (Linux)
sudo apt install python3.11 python3.11-venv

# Setup pipeline (auto-detects Python 3.11)
cd alt_ml_pipeline_1
bash scripts/setup_environment.sh

# Activate environment
source ~/venvs/alt_pipeline_1/bin/activate

# Test installation
python scripts/test_installation.py
```

---

**Recommended**: Use pyenv for clean Python version management across projects.
