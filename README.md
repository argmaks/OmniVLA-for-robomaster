# OmniVLA: An Omni-Modal Vision-Language-Action Model for Robot Navigation
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://omnivla-nav.github.io)


[Noriaki Hirose](https://sites.google.com/view/noriaki-hirose/)<sup>1, 2</sup>, [Catherine Glossop](https://catglossop.github.io/)<sup>1</sup>, [Dhruv Shah](https://robodhruv.github.io/)<sup>3</sup>, [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/)<sup>1</sup>

<sup>1</sup> UC Berkeley (_Berkeley AI Research_),  <sup>2</sup> Toyota Motor North America, ,  <sup>3</sup> Princeton University

IEEE International Conference on Robotics and Automation (ICRA) 2026

### Installation
Two equally supported installation paths are available — see [SETUP.md](SETUP.md) for full details.

**Option A — Pixi (recommended)**

```bash
# Server side (GPU cluster, CUDA 12.1+)
pixi install -e server && pixi run -e server setup

# Client side — pick the environment matching your ROS version:
#   client | client-humble | client-jazzy | client-kilted | client-noetic
pixi install -e client-humble
```

**Option B — Conda (manual)**

```bash
conda create -n omnivla python=3.11 -y && conda activate omnivla
pip install -e .
```

See [SETUP.md](SETUP.md) for the full conda setup (PyTorch, Flash Attention, etc.).

### Inference
1. Download our checkpoints and place them in our directory. "omnivla-original" is the trained checkpoints of the OmniVLA for paper submission. "omnivla-original-balance" contains the trained checkpoints of OmniVLA that account for the data balance in the LeLaN dataset. And "omnivla-finetuned-cast" is finetuned checkpoints with the [CAST](https://huggingface.co/datasets/catglossop/CAST-dataset) dataset.
    ```
    git clone https://huggingface.co/NHirose/omnivla-original
    git clone https://huggingface.co/NHirose/omnivla-original-balance    
    git clone https://huggingface.co/NHirose/omnivla-finetuned-cast
    ```
    > **Note:** `git clone` from HuggingFace uses Git LFS for large files. If the cloned `.safetensors` files are only ~135 bytes (LFS pointer stubs rather than real weights), loading the model will fail with `SafetensorError: header too large`. To download the actual weights, use `huggingface_hub` instead:
    > ```python
    > from huggingface_hub import snapshot_download
    > snapshot_download(repo_id='NHirose/omnivla-original', local_dir='./omnivla-original')
    > ```
    > Repeat for each checkpoint you need (`omnivla-original-balance`, `omnivla-finetuned-cast`). Each checkpoint is ~14 GB, so ensure you have sufficient disk space.
2. Run OmniVLA using a sample current image, goal images, GPS pose, and language prompt. You can view the generated trajectory in the output figure 1_ex.jpg.
    ```
    python inference/run_omnivla.py
    ```
3. Change the goal modality: by default, our code generates actions based on the language prompt. To use a different modality, you can modify the settings around line 560. 
    
4. Run OmniVLA to control the real robot. Modify "run_omnivla.py" to update the robot’s state (camera image, GPS signal) and adjust the goal information accordingly. Then, feed the generated velocity commands to your robot.

5. To try the finetuned checkpoints with the CAST dataset, update the path and step number in "InferenceConfig" within "run_omnivla.py".

### Inference: OmniVLA-edge
1. Download our checkpoints and place them in our directory. 
    ```
    git clone https://huggingface.co/NHirose/omnivla-edge
    ```
2. Run OmniVLA-edge using a sample current image, goal images, GPS pose, and language prompt. You can view the generated trajectory in the output figure 1_ex_omnivla_edge.jpg.
    ```
    python inference/run_omnivla_edge.py
    ```
3. Change the goal modality: by default, our code generates actions based on the language prompt. To use a different modality, you can modify the settings around line 425. 
    
4. Run OmniVLA to control the real robot. Modify "run_omnivla_edge.py" to update the robot’s state (camera image, GPS signal) and adjust the goal information accordingly. Then, feed the generated velocity commands to your robot.

### Training
We provide the training code along with a sample dataloader to help you quickly understand the required data loading structure. Since preparing the full training dataset is resource-intensive, we include this simplified code base for convenience.

1. Downloading MBRA project code base:
    ```
    cd ..
    git clone https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA.git
    ```
2. Downloading MBRA model:
    ```
    cd OmniVLA_internal
    git clone https://huggingface.co/NHirose/MBRA/
    ```
3. You can set the training or debugging mode at line 10 in vla-scripts/train_omnivla.py. Note that even in debugging mode, the code requires at least 20 GB of GPU memory (we use an NVIDIA RTX 4090).

4. You can configure visualization at line 11 in vla-scripts/train_omnivla.py. During training, it should be set to False.
    
5. Training our policy from OpenVLA checkpoints (Please fill X):
    ```
    torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/train_omnivla.py  --vla_path openvla/openvla-7b --dataset_name omnivla --num_images_in_input 2 --batch_size X --wandb_entity "X" --wandb_project "omnivla"
    ```
6. Finetuning our OmniVLA (Please fill X):
    ```
    torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/train_omnivla.py  --vla_path ./omnivla-original --dataset_name omnivla --num_images_in_input 2 --batch_size X --wandb_entity "X" --wandb_project "omnivla"
    ````
7. Memo finetuning our OmniVLA on our large navigation dataset:
    ```
    conda activate omnivla_2
    cd /media/noriaki/Noriaki_Data/OmniVLA
    torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train_omnivla_dataset.py  --vla_path ./omnivla-original --dataset_name omnivla --wandb_entity "noriaki-hirose"   --wandb_project "omnivla"
    ```

### Training with GNM, LeLaN, Frodobots, BDD and CAST datasets
We provide training code that supports multiple public datasets. Before following the full training process, please first ensure that you can run the example training with the sample dataloader.

1. Downloading all datasets from the original website. ([GNM](https://github.com/robodhruv/visualnav-transformer), [LeLaN](https://github.com/NHirose/learning-language-navigation), [Frodobots](https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA), [CAST](https://openvla-oft.github.io/)) Please verify that the downloaded datasets work properly in their original codebase, except BDD dataset. Note that please download the LeLaN dataset from [this link](https://huggingface.co/datasets/NHirose/LeLaN_dataset_NoMaD_traj/tree/main) instead of [the original link](https://drive.google.com/file/d/1IazHcIyPGO7ENswz8_sGCIGBXF8_sZJK/view). The updated dataset already includes the NoMaD trajectories used for collision-avoidance supervision, you no longer need to compute the NoMaD policy during training. Please carefully follow the usage procedure described in the [LeLaN codebase](https://github.com/NHirose/learning-language-navigation) when working with the dataset.
 
2. Downloading the modified BDD dataset with MBRA annotations from [here](https://huggingface.co/datasets/NHirose/BDD_OmniVLA) and extract it. The image sequences in the modified dataset remain subject to the [original BDD license](http://bdd-data.berkeley.edu/download.html), while the additional MBRA annotations are released under the MIT license.

3. Downloading the lerobot code base for the Frodobots dataset dataloader:
    ```
    git clone https://github.com/huggingface/lerobot.git 
    ```
4. Edit the data path in config_nav/mbra_and_dataset_config.yaml:

5. Training our policy from OpenVLA checkpoints (Please fill X):
    ```
    torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/train_omnivla_dataset.py  --vla_path ./omnivla-original --dataset_name omnivla --wandb_entity "X"   --wandb_project "omnivla"
    ```
       
In our training setup, we use 8 Nvidia H100 GPUs (80 GB each) across 8 nodes. The batch sizes are configured as [LeLaN, GNM, Frodobots, BDD] = [4, 1, 1, 1], with gradient accumulation set to 4 steps. When finetuning with CAST dataset, we set the batch size as [LeLaN, CAST, GNM, Frodobots, BDD] = [2, 2, 1, 1, 1]. To do so, you need to directly edit train_omnivla_dataset.py.
    
---

## Repository Structure (current)

```
OmniVLA/
├── inference/
│   ├── run_omnivla.py          # Standalone inference script + Inference class
│   ├── run_omnivla_edge.py     # Lightweight edge-model inference script
│   ├── model_omnivla_edge.py   # EfficientNet-based OmniVLA-edge architecture
│   ├── utils_policy.py         # Image/model utilities for edge model
│   ├── server.py               # FastAPI remote inference server
│   ├── client.py               # Laptop-side client for server
│   ├── current_img.jpg         # Sample current image for testing
│   └── goal_img.jpg            # Sample goal image for testing
├── control/
│   └── control.py              # ROS 2 robot controller (RoboMaster + OmniVLA server)
├── prismatic/                  # Model backbone library (from OpenVLA-OFT)
│   ├── extern/hf/              # HuggingFace-compatible model/processor/config wrappers
│   ├── models/
│   │   ├── action_heads.py     # L1RegressionActionHead_idcat — maps hidden states → actions
│   │   ├── projectors.py       # ProprioProjector — maps goal pose → LLM token
│   │   └── backbones/          # Vision + LLM backbone definitions
│   ├── training/train_utils.py # Action mask helpers (get_current_action_mask, etc.)
│   └── vla/
│       ├── action_tokenizer.py # Discretises continuous actions for the prompt
│       └── constants.py        # NUM_ACTIONS_CHUNK=8, ACTION_DIM=4, POSE_DIM=4
├── vla-scripts/                # Training entry points
│   ├── train_omnivla.py        # Train from OpenVLA checkpoint (sample dataloader)
│   └── train_omnivla_dataset.py# Train on full multi-dataset mixture
├── config_nav/                 # YAML configs for navigation/training hyperparameters
├── experiments/robot/
│   └── openvla_utils.py        # Legacy client helper (get_action_from_server)
├── omnivla-original/           # Checkpoint: base model (step 120000, ~14 GB)
├── omnivla-finetuned-cast/     # Checkpoint: CAST-finetuned (step 210000, ~14 GB)
└── pyproject.toml              # Package dependencies (includes fastapi, uvicorn, utm)
```

---

## OmniVLA — Model and Inference Pipeline

### What OmniVLA is

OmniVLA is a 7-billion-parameter Vision-Language-Action model for robot navigation.  It is built on top of [OpenVLA-OFT](https://openvla-oft.github.io/) (itself based on Llama-2 + SigLIP vision backbone) and adds three navigation-specific components:

- **Multi-modal goal conditioning** — the robot can be directed by any combination of: a language instruction, a goal image, a GPS pose, or a satellite map.  A single integer `modality_id` (0–8) encodes which combination is active at each inference step, allowing the same model to handle all cases.
- **`ProprioProjector`** — a small MLP that projects the 4-D goal pose `[Δy, −Δx, cos θ, sin θ]` (in normalised metric units relative to the robot) into one extra LLM token, appended to the vision patch sequence.
- **`L1RegressionActionHead_idcat`** — a small head that maps per-token hidden states from the LLM's last layer into a continuous 8-step trajectory, bypassing the LLM's discrete token vocabulary.

### Step-by-step inference pipeline (`inference/run_omnivla.py`)

#### 1. GPS → local goal pose
```
current GPS + compass  ──utm──►  (easting, northing)
goal GPS + compass     ──utm──►  (easting, northing)
                                        │
                          rotate to robot body frame
                                        │
               goal_pose_loc_norm = [Δy/0.1, −Δx/0.1, cos Δθ, sin Δθ]
```
`Δx` is forward, `Δy` is left, spacings are in 0.1 m units.  The pose is clamped if the goal is more than 30 m away.  **In language-only mode (`modality_id=7`) this tensor is still computed and fed to the model but gated out by the modality conditioning.**

#### 2. Prompt construction
```
"What action should the robot take to <language instruction>?"
```
The *ground-truth* action slot in the prompt is filled with dummy tokenised actions (random values); only the hidden states matter, not the token predictions.  The `ActionTokenizer` converts continuous action vectors to discrete token IDs for this slot.

#### 3. Image processing
Both the current RGB image and the goal RGB image are processed by `PrismaticImageProcessor` (SigLIP-style transforms, resized to 224×224).  The resulting tensors are concatenated along the channel dimension:
```
pixel_values shape: (B, 2×C, H, W)   — current image first, goal image second
```
Each image produces N vision patches; the total patch count is `2N + 1` (the `+1` is the goal-pose token from `ProprioProjector`).

#### 4. VLA forward pass  (`OpenVLAForActionPrediction_MMNv1`)
```
pixel_values  ──vision backbone──►  2N patch embeddings
goal_pose     ──ProprioProjector──► 1  pose  embedding
                                        │
               concatenate with text token embeddings
                                        │
                          Llama-2 LLM (7B params)
                                        │
                      last-layer hidden states  (B, seq_len, D)
```
`modality_id` is broadcast into the LLM via learned embeddings that condition each attention layer on which input modalities are active.

#### 5. Action head  (`L1RegressionActionHead_idcat`)
Hidden states at the positions corresponding to the 8-step action chunk are extracted:
```
actions_hidden_states: (B, 8×4, D)
        │
        ▼
L1RegressionActionHead_idcat
        │
predicted_actions: (B, 8, 4)   — in normalised units
```
Each of the 8 steps has 4 values: `[dx, dy, cos_heading, sin_heading]`.

#### 6. Waypoint → velocity (PD controller)
```
waypoint = predicted_actions[0][4]      # look-ahead index 4 (5th step ≈ 0.5 m out)
dx, dy = waypoint[:2] * 0.1            # scale from normalised units to metres

linear_vel  =  dx / DT                 # DT = 1/3 s
angular_vel =  arctan(dy / dx) / DT
```
Velocities are then clipped to `[0, 0.5]` m/s and `[−1, 1]` rad/s, and a second limiter enforces `maxv = 0.3 m/s`, `maxw = 0.3 rad/s` while preserving the linear/angular ratio.

#### 7. Output
```
(linear_vel: float, angular_vel: float)
```
Forward speed in m/s and turning rate in rad/s, ready to send to a differential-drive robot.

### Key constants (`prismatic/vla/constants.py`)
| Constant | Value | Meaning |
|---|---|---|
| `NUM_ACTIONS_CHUNK` | 8 | Steps in predicted trajectory |
| `ACTION_DIM` | 4 | `[dx, dy, cos_h, sin_h]` per step |
| `POSE_DIM` | 4 | Goal pose vector length |
| `ACTION_PROPRIO_NORMALIZATION_TYPE` | `BOUNDS_Q99` | Quantile normalisation |

### Modality IDs
| ID | Active inputs |
|---|---|
| 0 | satellite only |
| 1 | pose + satellite |
| 2 | satellite + image |
| 3 | all (pose + satellite + image) |
| 4 | pose only |
| 5 | pose + image |
| 6 | image only |
| 7 | language only |
| 8 | language + pose |

---

## Remote Inference Server

The full OmniVLA model requires ~20 GB GPU memory and runs on the compute cluster.  `inference/server.py` exposes a single HTTP endpoint so a laptop (or any client) can request velocity commands without SSHing in.

### Architecture
```
Laptop ──HTTP POST /act──► SSH tunnel ──► Cluster (FastAPI server, GPU)
                                                │
                                    load model once at startup
                                    one Inference instance per request
                                    return {linear_vel, angular_vel}
```

### Starting the server (on the cluster)
```bash
conda activate omnivla
cd ~/OmniVLA
python -m inference.server          # loads model, listens on 0.0.0.0:8777
```
The server prints `OmniVLA model loaded — server ready.` when the ~14 GB checkpoint has been loaded into GPU memory.  Model loading takes ~1–2 minutes; subsequent requests run in ~100–200 ms.

Alternatively:
```bash
uvicorn inference.server:app --host 0.0.0.0 --port 8777
```

### Opening the SSH tunnel (on the laptop)
```bash
ssh -L 8777:localhost:8777 <cluster-hostname>
```
Keep this terminal open.  All traffic to `localhost:8777` on the laptop is forwarded to port 8777 on the cluster.  The cluster uses Kerberos authentication (`GSSAPIAuthentication yes`, `GSSAPIDelegateCredentials yes`).

### Calling the server from the laptop
```python
from PIL import Image
from inference.client import get_action

result = get_action(
    Image.open("frame.jpg"),       # current camera image (PIL)
    "move toward the blue bin",    # language instruction
)
print(result["linear_vel"], result["angular_vel"])
```

The smoke-test in `inference/client.py` (`python inference/client.py`) runs this exact call against `inference/current_img.jpg`.

### API reference

**`POST /act`**

| Field | Type | Required | Description |
|---|---|---|---|
| `current_image` | string (base64) | yes | Robot's current camera frame |
| `lan_inst` | string | yes | Language navigation goal |
| `goal_image` | string (base64) | no | Goal-location image (image_goal mode) |
| `current_lat/lon/compass` | float | no | Robot GPS position in degrees |
| `goal_lat/lon/compass` | float | no | Goal GPS position in degrees |
| `lan_prompt` | bool | no | Default `true` |
| `pose_goal` | bool | no | Default `false` |
| `image_goal` | bool | no | Default `false` |
| `satellite` | bool | no | Default `false` |

Response: `{"linear_vel": float, "angular_vel": float}`

When `goal_image` is omitted the server uses a black dummy image of the same size as `current_image`; when GPS fields are omitted dummy coordinates are used.  In both cases the values are fed to the model but gated out by `modality_id`.

### Implementation notes for debugging

- **`inference/run_omnivla.py` — `Inference` class**: All model references (`vla`, `action_head`, `pose_projector`, `device_id`, `num_patches`) and modality flags are instance attributes (`self.*`), not module globals.  The server passes them in at construction time; the standalone script passes them from its `__main__` block.
- **`inference/server.py` — `_model_state` dict**: The model is loaded once in the `startup` event handler and stored here.  A new `Inference` instance is created for each request (stateless — only the model handles are shared).
- **Visualization side-effect**: `run_omnivla()` always calls `save_robot_behavior()`, writing a plot to `inference/{count_id}_ex.jpg`.  In server mode `count_id` resets to 0 for each request, so `inference/0_ex.jpg` is overwritten every call.
- **Waypoints coordinate frame**: Robot body frame, X forward, Y left, in normalised 0.1 m units.  The velocity controller picks step index 4 as its look-ahead target.
- **`modality_id=7` (language only)**: The goal image tensor (black dummy) and goal pose tensor (dummy GPS) are still passed to the model but the modality conditioning causes them to be ignored.  If you see unexpected behaviour in language-only mode, check that `lan_prompt=True` and the other three flags are `False`.

---

## ROS 2 Robot Controller (`control/control.py`)

`control/control.py` closes the loop between the OmniVLA inference server and a physical RoboMaster robot running ROS 2 Humble.  It subscribes to the robot's compressed camera stream, sends each frame to the server, and publishes the returned velocity command to `cmd_vel` at 10 Hz (so the firmware watchdog is always satisfied regardless of inference latency).

### Prerequisites

- SSH tunnel open to the cluster (see [Opening the SSH tunnel](#opening-the-ssh-tunnel-on-the-laptop) above).
- `pixi install -e client-humble` completed.
- ROS 2 environment active and the robot reachable (`ros2 topic list` shows `/robomaster_10/…`).

### Running the controller

```bash
# Default instruction ("move forward")
pixi run -e client-humble python control/control.py

# Custom instruction
pixi run -e client-humble python control/control.py "move toward the blue bin"

# Custom server URL
pixi run -e client-humble python control/control.py --server http://localhost:8777 "navigate to the door"
```

Press **Ctrl-C** to stop — the controller publishes a zero-velocity command before exiting.

### Topics

| Direction | Topic | Type |
|---|---|---|
| Subscribe | `/robomaster_10/camera_0/image_raw/compressed` | `sensor_msgs/CompressedImage` |
| Publish   | `/robomaster_10/cmd_vel` | `geometry_msgs/Twist` |

### CLI reference

```
usage: control.py [-h] [--server URL] [--camera-topic TOPIC] [instruction]

positional arguments:
  instruction           Language navigation goal (default: "move forward")

options:
  --server URL          OmniVLA server URL (default: http://localhost:8777)
  --camera-topic TOPIC  ROS 2 compressed image topic
```

### Acknowledgement
We implement our ideas and design choices on top of the pretrained checkpoints. Our work builds upon the [OpenVLA-OFT](https://openvla-oft.github.io/) codebase, with additional code added to create OmniVLA. As such, our implementation leverages many components of the OpenVLA-OFT codebase. We sincerely appreciate the effort and contributions of the OpenVLA-OFT team!

## Citing
```
@misc{hirose2025omnivla,
      title={OmniVLA: An Omni-Modal Vision-Language-Action Model for Robot Navigation}, 
      author={Noriaki Hirose and Catherine Glossop and Dhruv Shah and Sergey Levine},
      year={2025},
      eprint={2509.19480},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.19480}, 
}
```
