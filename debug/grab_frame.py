#!/usr/bin/env python3
"""Grab one frame from the robot camera and save it to debug/current.jpg."""

import io
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from PIL import Image

CAMERA_TOPIC = "/robomaster_10/camera_0/image_raw/compressed"
OUT_PATH = os.path.join(os.path.dirname(__file__), "current.jpg")


class _Grabber(Node):
    def __init__(self):
        super().__init__("frame_grabber")
        self.create_subscription(CompressedImage, CAMERA_TOPIC, self._cb, qos_profile_sensor_data)
        self.get_logger().info(f"Waiting for a frame on {CAMERA_TOPIC} ...")

    def _cb(self, msg: CompressedImage):
        img = Image.open(io.BytesIO(bytes(msg.data))).convert("RGB")
        img.save(OUT_PATH)
        self.get_logger().info(f"Saved {img.size[0]}x{img.size[1]} frame to {OUT_PATH}")
        raise SystemExit


def main():
    rclpy.init()
    node = _Grabber()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
