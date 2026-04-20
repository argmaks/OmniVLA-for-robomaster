#!/usr/bin/env python3
"""
OmniVLA ROS 2 controller for the RoboMaster platform.

Subscribes to the compressed camera topic, sends each frame to the OmniVLA
remote inference server, and publishes the returned velocity command to cmd_vel.

Run inside the client-humble pixi environment:
    pixi run -e client-humble python control/control.py

Environment variables (all optional):
    OMNIVLA_SERVER_URL   OmniVLA server URL  (default: http://localhost:8777)
"""

import argparse
import base64
import io
import os
import signal
import threading

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

import requests
from PIL import Image as PILImage

# ── Configuration ──────────────────────────────────────────────────────────────

SERVER_URL   = os.environ.get("OMNIVLA_SERVER_URL", "http://localhost:8777")
ROBOT_NS     = "robomaster_10"
CAMERA_TOPIC = f"/{ROBOT_NS}/camera_0/image_raw/compressed"

DEFAULT_INSTRUCTION = "move forward"

# How often to publish the current command to satisfy the firmware watchdog
PUBLISH_HZ = 10

# ── Helpers ────────────────────────────────────────────────────────────────────

def _jpeg_to_b64(jpeg_bytes: bytes) -> str:
    return base64.b64encode(jpeg_bytes).decode()


def _compressed_to_pil(msg: CompressedImage) -> PILImage.Image:
    return PILImage.open(io.BytesIO(bytes(msg.data))).convert("RGB")


def query_server(server_url: str, image: PILImage.Image, instruction: str, timeout: float = 30.0) -> dict:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    payload = dict(
        current_image=_jpeg_to_b64(buf.getvalue()),
        lan_inst=instruction,
        lan_prompt=True,
        pose_goal=False,
        image_goal=False,
        satellite=False,
    )
    resp = requests.post(f"{server_url}/act", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ── ROS 2 Node ─────────────────────────────────────────────────────────────────

class OmniVLAController(Node):
    def __init__(self, instruction: str, server_url: str, camera_topic: str):
        super().__init__("omnivla_controller")

        self._instruction = instruction
        self._server_url  = server_url
        self._running     = True

        self._latest_image: PILImage.Image | None = None
        self._image_lock = threading.Lock()

        self._current_cmd = Twist()
        self._cmd_lock = threading.Lock()

        cmd_vel_topic = f"/{ROBOT_NS}/cmd_vel"

        self._pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.create_subscription(CompressedImage, camera_topic, self._image_cb, qos_profile_sensor_data)
        self.create_timer(1.0 / PUBLISH_HZ, self._publish_cb)

        self._infer_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="omnivla_infer"
        )
        self._infer_thread.start()

        self.get_logger().info("OmniVLA controller ready")
        self.get_logger().info(f"  namespace   : {ROBOT_NS}")
        self.get_logger().info(f"  cmd_vel     : {cmd_vel_topic}")
        self.get_logger().info(f"  camera      : {camera_topic}")
        self.get_logger().info(f"  server      : {server_url}")
        self.get_logger().info(f"  instruction : {instruction}")

    def _image_cb(self, msg: CompressedImage):
        try:
            pil_img = _compressed_to_pil(msg)
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return
        with self._image_lock:
            self._latest_image = pil_img

    def _publish_cb(self):
        with self._cmd_lock:
            cmd = self._current_cmd
        self._pub.publish(cmd)

    def _inference_loop(self):
        wait = threading.Event()
        while self._running:
            with self._image_lock:
                image = self._latest_image

            if image is None:
                self.get_logger().info("Waiting for camera image …", throttle_duration_sec=5.0)
                wait.wait(timeout=0.5)
                continue

            try:
                result = query_server(self._server_url, image, self._instruction)
                linear_vel  = float(result["linear_vel"])
                angular_vel = float(result["angular_vel"])

                cmd = Twist()
                cmd.linear.x  = linear_vel
                cmd.angular.z = angular_vel

                self.get_logger().info(
                    f"vx={linear_vel:.3f} m/s  wz={angular_vel:.3f} rad/s"
                )
            except Exception as e:
                self.get_logger().error(f"Inference error: {e}")
                cmd = Twist()

            with self._cmd_lock:
                self._current_cmd = cmd

            wait.wait(timeout=0.05)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OmniVLA ROS 2 robot controller")
    parser.add_argument(
        "instruction",
        nargs="?",
        default=DEFAULT_INSTRUCTION,
        help=f'Language navigation goal (default: "{DEFAULT_INSTRUCTION}")',
    )
    parser.add_argument(
        "--server",
        default=SERVER_URL,
        help=f"OmniVLA server URL (default: {SERVER_URL})",
    )
    parser.add_argument(
        "--camera-topic",
        default=CAMERA_TOPIC,
        help=f"ROS 2 compressed image topic (default: {CAMERA_TOPIC})",
    )
    args = parser.parse_args()

    rclpy.init()
    node = OmniVLAController(
        instruction=args.instruction,
        server_url=args.server,
        camera_topic=args.camera_topic,
    )

    def _sigint(sig, frame):
        node._running = False
        node._pub.publish(Twist())
        rclpy.shutdown()

    signal.signal(signal.SIGINT, _sigint)

    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
