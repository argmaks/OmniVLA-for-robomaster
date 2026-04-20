# ===============================================================
# OmniVLA Remote Inference Server
# ===============================================================
# Exposes POST /act — accepts a current image and a language
# instruction; returns a single linear/angular velocity command.
#
# Start with:
#   python -m inference.server
# or:
#   uvicorn inference.server:app --host 0.0.0.0 --port 8777
#
# Then SSH-tunnel from your laptop:
#   ssh -L 8777:localhost:8777 <cluster-host>
# ===============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import base64
import io
import logging
import math
from typing import Optional

logger = logging.getLogger("omnivla.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

import numpy as np
import utm
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from inference.run_omnivla import Inference, InferenceConfig, define_model

# ---------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------

# Dummy GPS used when pose/satellite modalities are not requested.
# The model receives the tensor but modality_id gates it out.
_DUMMY_LAT, _DUMMY_LON, _DUMMY_COMPASS = 37.87371258374039, -122.26729417226024, 0.0


class ActRequest(BaseModel):
    # --- Required ---
    current_image: str   # base64-encoded JPEG/PNG
    lan_inst: str        # language goal, e.g. "move toward the blue bin"

    # --- Optional: non-language modalities ---
    # goal_image: base64 image of the goal location (for image_goal mode)
    goal_image: Optional[str] = None
    # GPS (degrees) for pose_goal / satellite modes
    current_lat: Optional[float] = None
    current_lon: Optional[float] = None
    current_compass: Optional[float] = None   # degrees
    goal_lat: Optional[float] = None
    goal_lon: Optional[float] = None
    goal_compass: Optional[float] = None      # degrees
    # Modality flags — default is language-only (modality_id = 7)
    lan_prompt: bool = True
    pose_goal: bool = False
    image_goal: bool = False
    satellite: bool = False


class ActResponse(BaseModel):
    linear_vel: float    # m/s, forward, clipped to [0, 0.3]
    angular_vel: float   # rad/s, clipped to [-1, 1]


# ---------------------------------------------------------------
# App + global model state
# ---------------------------------------------------------------

app = FastAPI(title="OmniVLA Inference Server")

_model_state: dict = {}


def _decode_image(b64_str: str) -> Image.Image:
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data)).convert("RGB")


@app.on_event("startup")
def load_model():
    cfg = InferenceConfig()
    vla, action_head, pose_projector, device_id, num_patches, action_tokenizer, processor = define_model(cfg)
    _model_state.update(
        vla=vla,
        action_head=action_head,
        pose_projector=pose_projector,
        device_id=device_id,
        num_patches=num_patches,
        action_tokenizer=action_tokenizer,
        processor=processor,
    )
    print("OmniVLA model loaded — server ready.")


# ---------------------------------------------------------------
# Inference endpoint
# ---------------------------------------------------------------

@app.post("/act", response_model=ActResponse)
def act(req: ActRequest):
    if not _model_state:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        current_image_pil = _decode_image(req.current_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"current_image decode error: {e}")

    logger.info(
        f"[API /act] current_image decoded: {current_image_pil.size} WxH, mode={current_image_pil.mode}"
    )
    logger.info(f"[API /act] language instruction : {req.lan_inst!r}")
    logger.info(
        f"[API /act] modality flags: lan_prompt={req.lan_prompt}, pose_goal={req.pose_goal}, "
        f"image_goal={req.image_goal}, satellite={req.satellite}"
    )

    # Goal image: use a black dummy when running in language-only mode
    if req.goal_image is not None:
        try:
            goal_image_pil = _decode_image(req.goal_image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"goal_image decode error: {e}")
    else:
        goal_image_pil = Image.new("RGB", current_image_pil.size, (0, 0, 0))

    logger.info(
        f"[API /act] goal_image: {goal_image_pil.size} WxH, mode={goal_image_pil.mode} "
        f"(provided={req.goal_image is not None})"
    )

    # GPS: use dummy values when not provided (gated out by modality_id)
    current_lat = req.current_lat if req.current_lat is not None else _DUMMY_LAT
    current_lon = req.current_lon if req.current_lon is not None else _DUMMY_LON
    current_compass = req.current_compass if req.current_compass is not None else _DUMMY_COMPASS
    goal_lat = req.goal_lat if req.goal_lat is not None else _DUMMY_LAT
    goal_lon = req.goal_lon if req.goal_lon is not None else _DUMMY_LON
    goal_compass_deg = req.goal_compass if req.goal_compass is not None else _DUMMY_COMPASS

    goal_utm = utm.from_latlon(goal_lat, goal_lon)
    goal_compass_rad = -float(goal_compass_deg) / 180.0 * math.pi

    inference = Inference(
        save_dir="./inference",
        lan_inst_prompt=req.lan_inst,
        goal_utm=goal_utm,
        goal_compass=goal_compass_rad,
        goal_image_PIL=goal_image_pil,
        action_tokenizer=_model_state["action_tokenizer"],
        processor=_model_state["processor"],
        vla=_model_state["vla"],
        action_head=_model_state["action_head"],
        pose_projector=_model_state["pose_projector"],
        device_id=_model_state["device_id"],
        num_patches=_model_state["num_patches"],
        satellite=req.satellite,
        pose_goal=req.pose_goal,
        image_goal=req.image_goal,
        lan_prompt=req.lan_prompt,
    )

    linear_vel, angular_vel = inference.run_omnivla(
        current_image_pil,
        current_lat,
        current_lon,
        current_compass,
    )

    return ActResponse(
        linear_vel=float(linear_vel),
        angular_vel=float(angular_vel),
    )


if __name__ == "__main__":
    uvicorn.run("inference.server:app", host="0.0.0.0", port=8777, reload=False)
