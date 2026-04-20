#!/usr/bin/env python3
"""
Arrow-key teleoperation for the RoboMaster via ROS 2.

  ↑ / ↓   — forward / backward  (LINEAR_VEL m/s)
  ← / →   — turn left / right   (ANGULAR_VEL rad/s)
  Combinations work simultaneously. Releasing a key stops that motion.
  Esc / q — quit

Requires pynput (real key-down/up events):
    pip install pynput

Run inside the client-humble pixi environment:
    pixi run -e client-humble python debug/teleop.py
"""

import signal
import sys
import threading

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from pynput import keyboard as kb

# ── Tuning ─────────────────────────────────────────────────────────────────────
LINEAR_VEL  = 0.5   # m/s
ANGULAR_VEL = 0.75   # rad/s
PUBLISH_HZ  = 10

ROBOT_NS  = "robomaster_10"
CMD_TOPIC = f"/{ROBOT_NS}/cmd_vel"

# ── ROS 2 node ─────────────────────────────────────────────────────────────────

class TeleopNode(Node):
    def __init__(self):
        super().__init__("teleop_keyboard")
        self._pub = self.create_publisher(Twist, CMD_TOPIC, 10)
        self.create_timer(1.0 / PUBLISH_HZ, self._publish_cb)

        self._held: set[str] = set()
        self._lock = threading.Lock()

        self.get_logger().info("Teleop ready — arrow keys to move, Esc/q to quit")
        self.get_logger().info(f"  cmd_vel : {CMD_TOPIC}")
        self.get_logger().info(f"  linear  : ±{LINEAR_VEL} m/s   angular: ±{ANGULAR_VEL} rad/s")

    def key_down(self, direction: str):
        with self._lock:
            self._held.add(direction)

    def key_up(self, direction: str):
        with self._lock:
            self._held.discard(direction)

    def _publish_cb(self):
        with self._lock:
            held = set(self._held)

        lin  =  LINEAR_VEL  if "fwd"   in held else 0.0
        lin -= (LINEAR_VEL  if "back"  in held else 0.0)
        ang  =  ANGULAR_VEL if "left"  in held else 0.0
        ang -= (ANGULAR_VEL if "right" in held else 0.0)

        msg = Twist()
        msg.linear.x  = lin
        msg.angular.z = (-1) * ang
        self._pub.publish(msg)

        sys.stdout.write(f"\r  vx={lin:+.2f} m/s   wz={ang:+.2f} rad/s    ")
        sys.stdout.flush()

    def stop(self):
        self._pub.publish(Twist())


# ── pynput key listener ────────────────────────────────────────────────────────

_KEY_MAP = {
    kb.Key.up:    "fwd",
    kb.Key.down:  "back",
    kb.Key.left:  "left",
    kb.Key.right: "right",
}


def run_listener(node: TeleopNode, stop_event: threading.Event):
    def on_press(key):
        direction = _KEY_MAP.get(key)
        if direction:
            node.key_down(direction)
            return
        # quit on Esc or q
        char = getattr(key, "char", None)
        if key == kb.Key.esc or char in ("q", "Q"):
            stop_event.set()
            return False  # stops the listener

    def on_release(key):
        direction = _KEY_MAP.get(key)
        if direction:
            node.key_up(direction)

    with kb.Listener(on_press=on_press, on_release=on_release) as listener:
        stop_event.wait()   # block until quit is requested
        listener.stop()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = TeleopNode()

    stop_event = threading.Event()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    def _sigint(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _sigint)

    print("\n  ↑/↓ forward/backward    ←/→ turn    Esc/q quit\n")

    try:
        run_listener(node, stop_event)
    finally:
        node.stop()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
