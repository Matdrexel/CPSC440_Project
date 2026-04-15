# Complete Poker AI Project Setup Guide

This guide will take you from an empty folder to a fully working development environment with PokerKit and OpenSpiel.

---

## Prerequisites

Make sure you have these installed:
- **Python 3.11+** (required for PokerKit)
- **Git** (for version control)
- **VSCode** (you already have this)

### Check if Python is installed:
```bash
python --version
# or
python3 --version
```

If not installed, download from: https://www.python.org/downloads/

### Check if Git is installed:
```bash
git --version
```

If not installed:
- **Windows**: https://git-scm.com/download/win
- **Mac**: `brew install git` or download from https://git-scm.com/download/mac
- **Linux**: `sudo apt install git`

---

## Step 1: Clone Your Friend's Repository

Open a terminal in VSCode (Terminal → New Terminal) or use your system terminal.

```bash
# Navigate to where you want your project (e.g., your Documents folder)
cd ~/Documents

# Clone the repository
git clone https://github.com/Matdrexel/CPSC440_Project.git

# Enter the project directory
cd CPSC440_Project
```

---

## Step 2: Create a Python Virtual Environment

A virtual environment keeps your project dependencies isolated.

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt now.

---

## Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install PokerKit
pip install pokerkit

# Install OpenSpiel (may take a few minutes)
pip install open_spiel

# Install other useful packages
pip install numpy pandas matplotlib jupyter

# Save dependencies to requirements.txt
pip freeze > requirements.txt
```

---

## Step 4: Create the Project Structure

Your folder structure should look like this:

```
CPSC440_Project/
├── .vscode/
│   └── settings.json
├── .gitignore
├── requirements.txt
├── README.md
├── gameplay/           # Your friend's existing folder
├── pokerkit_examples/  # New: PokerKit testing
│   ├── __init__.py
│   ├── basic_game.py
│   ├── hand_history_demo.py
│   └── simulate_games.py
├── openspiel_examples/ # New: OpenSpiel testing
│   ├── __init__.py
│   ├── kuhn_poker.py
│   └── cfr_training.py
├── agents/             # New: Your AI agents
│   ├── __init__.py
│   ├── random_agent.py
│   └── simple_agent.py
├── data/               # New: Store hand histories
│   └── .gitkeep
└── notebooks/          # New: Jupyter notebooks for experiments
    └── exploration.ipynb
```

Create these folders:
```bash
mkdir -p pokerkit_examples openspiel_examples agents data notebooks
touch pokerkit_examples/__init__.py openspiel_examples/__init__.py agents/__init__.py
touch data/.gitkeep
```

---

## Step 5: VSCode Setup

### Install recommended extensions:
1. Open VSCode Extensions (Ctrl+Shift+X or Cmd+Shift+X)
2. Search and install:
   - **Python** (Microsoft)
   - **Pylance** (Microsoft)
   - **Jupyter** (Microsoft)
   - **GitLens** (optional but helpful)

### Select Python interpreter:
1. Press Ctrl+Shift+P (or Cmd+Shift+P on Mac)
2. Type "Python: Select Interpreter"
3. Choose the one in your venv folder (e.g., `./venv/bin/python`)

---

## Step 6: Verify Installation

Create a quick test file to verify everything works:

```bash
# Create test file
touch test_installation.py
```

Add this content to `test_installation.py`:

```python
# test_installation.py
print("Testing installations...\n")

# Test PokerKit
try:
    import pokerkit
    print(f"✓ PokerKit installed: version info available")
    from pokerkit import NoLimitTexasHoldem, Automation
    print("✓ PokerKit imports work")
except ImportError as e:
    print(f"✗ PokerKit error: {e}")

# Test OpenSpiel
try:
    import pyspiel
    print(f"✓ OpenSpiel installed")
    game = pyspiel.load_game("kuhn_poker")
    print(f"✓ OpenSpiel can load games: {game}")
except ImportError as e:
    print(f"✗ OpenSpiel error: {e}")

print("\n✓ All tests passed! You're ready to go.")
```

Run it:
```bash
python test_installation.py
```

---

## Step 7: Git Configuration

```bash
# Set your identity (use your own info)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Check remote is set correctly
git remote -v
```

---

## Quick Reference Commands

```bash
# Activate virtual environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Run a Python file
python filename.py

# Install a new package
pip install package_name
pip freeze > requirements.txt  # Update requirements

# Git commands
git status                    # Check what's changed
git add .                     # Stage all changes
git commit -m "Your message"  # Commit changes
git push                      # Push to GitHub
git pull                      # Get latest from GitHub
```

---

## Troubleshooting

### "python not found"
Try `python3` instead of `python`, or reinstall Python and make sure to check "Add to PATH" during installation.

### OpenSpiel installation fails
OpenSpiel can be tricky. Try:
```bash
pip install open_spiel --no-cache-dir
```

If that fails on Windows, you may need Visual Studio Build Tools.

### Permission denied
On Mac/Linux, you might need:
```bash
chmod +x venv/bin/activate
```

### VSCode doesn't see packages
Make sure you selected the correct Python interpreter (the one in your venv).
