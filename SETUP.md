# Setup Instructions

## Set Up Conda Environment

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
