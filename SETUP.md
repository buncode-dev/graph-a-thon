# Development Environment Setup Guide

This guide walks you through setting up your programming environment from scratch. By the end, you'll have Git installed and configured, a text editor ready to go, the repository cloned, and `uv` managing your Python packages.

---

## 1. Install Git

### macOS

Open Terminal and install Git via Homebrew. If you don't have Homebrew, install it first:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install Git:

```bash
brew install git
```

Alternatively, running `git --version` in Terminal will prompt you to install the Xcode Command Line Tools, which include Git.

### Windows

Download the installer from [git-scm.com](https://git-scm.com/download/win) and run it. The default options work well for most users. Make sure **"Git from the command line and also from 3rd-party software"** is selected during installation so that `git` is available in your terminal.

After installation, open **Git Bash** or **PowerShell** and verify:

```bash
git --version
```

> **Troubleshooting PATH on Windows:** The Git installer and `uv` installer both add themselves to your system PATH automatically. If `git` or `uv` isn't recognized after installation, close and reopen your terminal first. If it still doesn't work, you can add them manually: open **Settings → System → About → Advanced system settings → Environment Variables**, find `Path` under "User variables", click **Edit**, and add the relevant directory (typically `C:\Program Files\Git\cmd` for Git, and `%USERPROFILE%\.local\bin` for `uv`).

### Linux

Use your distribution's package manager:

```bash
# Debian / Ubuntu
sudo apt update && sudo apt install git

# Fedora
sudo dnf install git

# Arch
sudo pacman -S git
```

---

## 2. Configure Your Git Identity

Before you can make commits, Git needs to know who you are. Run these commands in your terminal (use the email associated with your Git hosting account):

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

You can verify your configuration at any time:

```bash
git config --list
```

---

## 3. Create a GitHub Account

If you don't already have one, sign up at [github.com](https://github.com). Use the same email you configured in the previous step.

### Set Up SSH Authentication (Recommended)

SSH keys let you push and pull without entering your password each time.

**Generate a key:**

```bash
ssh-keygen -t ed25519 -C "you@example.com"
```

Press Enter to accept the default file location, then set a passphrase (or leave it blank).

**Copy the public key:**

```bash
# macOS
pbcopy < ~/.ssh/id_ed25519.pub

# Linux (requires xclip)
xclip -selection clipboard < ~/.ssh/id_ed25519.pub

# Windows (Git Bash)
cat ~/.ssh/id_ed25519.pub | clip
```

**Add it to your account:** Go to **GitHub → Settings → SSH and GPG keys → New SSH key**, paste the key, and save.

**Test the connection:**

```bash
ssh -T git@github.com
```

You should see a message confirming successful authentication.

---

## 4. Install a Text Editor

Pick whichever editor you prefer. Both are excellent, free, and support extensions for Python development.

### Option A: Visual Studio Code

VS Code is the most popular choice with a massive extension ecosystem.

| Platform | Installation |
|----------|-------------|
| **macOS** | Download from [code.visualstudio.com](https://code.visualstudio.com) or run `brew install --cask visual-studio-code` |
| **Windows** | Download the installer from [code.visualstudio.com](https://code.visualstudio.com) |
| **Linux** | Download the `.deb` or `.rpm` from [code.visualstudio.com](https://code.visualstudio.com), or install via snap: `sudo snap install code --classic` |

**Recommended extensions:** Open VS Code and install these from the Extensions panel (`Ctrl+Shift+X` / `Cmd+Shift+X`):

- **Python** (by Microsoft) — IntelliSense, linting, debugging
- **Ruff** — fast Python linter and formatter
- **GitLens** — enhanced Git integration

### Option B: Zed

Zed is a newer, high-performance editor built in Rust with first-class support for collaborative editing.

| Platform | Installation |
|----------|-------------|
| **macOS** | Download from [zed.dev](https://zed.dev) or run `brew install --cask zed` |
| **Windows** | Download from [zed.dev/download](https://zed.dev/download) (currently in preview) |
| **Linux** | Install via the official script: `curl -fsSL https://zed.dev/install.sh \| sh` |

Zed has built-in support for Python syntax highlighting and LSP integration. You can configure language servers through Zed's settings (`Cmd+,` / `Ctrl+,`) under the `lsp` section.

---

## 5. Install uv

`uv` is an extremely fast Python package and project manager written in Rust. It replaces `pip`, `venv`, `pip-tools`, and more.

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, restart your shell or run the command printed by the installer to add `uv` to your PATH.

### Windows

Open PowerShell and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, **close and reopen your terminal** for the PATH changes to take effect.

### Verify Installation

```bash
uv --version
```

---

## 6. Clone the Repository

Navigate to the directory where you want the project to live, then clone it:

```bash
# Using SSH (recommended if you set up SSH keys)
git clone git@github.com:Meepst/graph-a-thon.git

# Using HTTPS (if you prefer token-based auth)
git clone https://github.com/Meepst/graph-a-thon.git
```

Then move into the project directory:

```bash
cd graph-a-thon
```

---

## 7. Set Up the Project with uv

### Install Dependencies

This project uses a `pyproject.toml` to manage dependencies. Simply run:

```bash
uv sync
```

This will automatically install the correct Python version (3.13, as specified in `.python-version`), create a `.venv` in the project directory, and install all dependencies in one step.

If you need to install Python 3.13 explicitly first:

```bash
uv python install 3.13
```

### Adding Packages

```bash
# Add a dependency to the project
uv add requests

# Add a dev-only dependency
uv add --dev pytest
```

### Running Scripts

You can run Python scripts through the project environment without activating it:

```bash
uv run python main.py
uv run pytest
```

---

## 8. Open the Project in Your Editor

### VS Code

```bash
code .
```

VS Code should detect the `.venv` automatically. If it doesn't, open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`), search for **"Python: Select Interpreter"**, and point it to `.venv/bin/python` (or `.venv\Scripts\python.exe` on Windows).

### Zed

```bash
zed .
```

Zed will pick up the virtual environment from the project directory.

---

## Quick Reference

| Task | Command |
|------|---------|
| Check Git version | `git --version` |
| Check uv version | `uv --version` |
| Install dependencies | `uv sync` |
| Add a package | `uv add <package>` |
| Run a script | `uv run python <script>.py` |
| Pull latest changes | `git pull` |
| Check repo status | `git status` |

---

You're all set. If you run into issues, check the [uv documentation](https://docs.astral.sh/uv/) or the [Git documentation](https://git-scm.com/doc).
