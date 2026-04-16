# Setup Instructions

Two equally supported installation paths are available: **pixi** (recommended, reproduces exact dependencies) and **conda** (manual, flexible).

---

## Option A — Pixi (recommended)

[Pixi](https://prefix.dev/docs/pixi/overview) manages separate locked environments for the GPU inference server and the robot client.  Install pixi once, then use it for both sides.

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash
```

### Server side (GPU cluster)

Requires an NVIDIA GPU with CUDA 12.1+ drivers.

```bash
# Install all locked server dependencies
pixi install -e server

# Install git-sourced forks + editable OmniVLA package
# (run once after `pixi install -e server`)
pixi run -e server setup

# Start the inference server on port 8777
pixi run -e server start
```

> **Flash Attention**: the `setup` task intentionally skips flash-attn by default (the relevant line is commented out in `pixi.toml`).  To enable it, uncomment the `flash-attn` line in the `setup` task and re-run `pixi run -e server setup`.

### Client side (robot / laptop)

Choose the environment that matches your ROS version.  On macOS (dev / smoke-testing) use the plain `client` environment.

| Environment | Use case |
|---|---|
| `client` | No ROS — macOS or bare Linux, local dev |
| `client-humble` | ROS 2 Humble (Ubuntu 22.04) |
| `client-jazzy` | ROS 2 Jazzy (Ubuntu 24.04) |
| `client-kilted` | ROS 2 Kilted |
| `client-noetic` | ROS 1 Noetic (legacy robots) |

```bash
# Example: ROS 2 Humble robot
pixi install -e client-humble
```

For environments with a ROS activation script (`install/setup.bash`), pixi sources it automatically on shell entry — no manual `source` needed.

---

## Option B — Conda (manual)

```bash
# Create and activate conda environment
conda create -n omnivla python=3.11 -y
conda activate omnivla

# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
pip3 install numpy torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# Clone openvla-oft repo and pip install to download dependencies
git clone https://github.com/NHirose/OmniVLA.git
cd OmniVLA
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```
