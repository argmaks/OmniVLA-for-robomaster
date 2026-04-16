# ===============================================================
# OmniVLA Remote Inference Client
# ===============================================================
# Run on your laptop after opening an SSH tunnel to the cluster:
#
#   ssh -L 8777:localhost:8777 <cluster-host>
#
# Minimal usage (language-only mode):
#
#   from inference.client import get_action
#   from PIL import Image
#
#   result = get_action(Image.open("frame.jpg"), "move toward the blue bin")
#   # result["linear_vel"], result["angular_vel"] → send to robot
# ===============================================================

import base64
import io
from typing import Optional

import requests
from PIL import Image


SERVER_URL = "http://localhost:8777"


def _encode_image(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def get_action(
    current_image: Image.Image,
    lan_inst: str,
    *,
    # Non-language modalities — omit for language-only (default)
    goal_image: Optional[Image.Image] = None,
    current_lat: Optional[float] = None,
    current_lon: Optional[float] = None,
    current_compass: Optional[float] = None,
    goal_lat: Optional[float] = None,
    goal_lon: Optional[float] = None,
    goal_compass: Optional[float] = None,
    lan_prompt: bool = True,
    pose_goal: bool = False,
    image_goal: bool = False,
    satellite: bool = False,
    server_url: str = SERVER_URL,
    timeout: float = 30.0,
) -> dict:
    """
    Send one inference request to the OmniVLA server; return velocity commands.

    Required:
        current_image  PIL image from the robot's camera.
        lan_inst       Language goal, e.g. "move toward the blue bin".

    Returns:
        {"linear_vel": float, "angular_vel": float}
        linear_vel  — forward speed in m/s, clipped to [0, 0.3]
        angular_vel — turning rate in rad/s, clipped to [-1, 1]
    """
    payload: dict = dict(
        current_image=_encode_image(current_image),
        lan_inst=lan_inst,
        lan_prompt=lan_prompt,
        pose_goal=pose_goal,
        image_goal=image_goal,
        satellite=satellite,
    )
    if goal_image is not None:
        payload["goal_image"] = _encode_image(goal_image)
    if current_lat is not None:
        payload["current_lat"] = current_lat
    if current_lon is not None:
        payload["current_lon"] = current_lon
    if current_compass is not None:
        payload["current_compass"] = current_compass
    if goal_lat is not None:
        payload["goal_lat"] = goal_lat
    if goal_lon is not None:
        payload["goal_lon"] = goal_lon
    if goal_compass is not None:
        payload["goal_compass"] = goal_compass

    resp = requests.post(f"{server_url}/act", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------
# Quick smoke-test — language-only mode
# ---------------------------------------------------------------
if __name__ == "__main__":
    import sys, os

    current_img_path = os.path.join(os.path.dirname(__file__), "current_img.jpg")
    if not os.path.exists(current_img_path):
        sys.exit("Place current_img.jpg in the inference/ directory first.")

    result = get_action(
        Image.open(current_img_path),
        "move toward the blue trash bin",
    )
    print("linear_vel :", result["linear_vel"])
    print("angular_vel:", result["angular_vel"])
