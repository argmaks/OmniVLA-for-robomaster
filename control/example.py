#!/usr/bin/env python3
"""
VLA (Vision-Language-Action) controller for the RoboMaster platform.

The VLM is shown a camera frame and asked to output a *semantic action*
from a fixed vocabulary (forward, backward, left, right, rotate_left,
rotate_right, stop) together with a magnitude (0–1) and a short reason.
Code maps these to a Twist message. This is more robust than asking the
model to output raw velocities:
  • Fewer output fields → fewer parse failures
  • Whitelisted actions → easy to validate
  • Magnitude gives continuous control
  • "reason" field is free text → useful for debugging

Action → Twist mapping (actual hardware signs; robot behaves as ROS REP-103 Z-up):
  forward       linear_x = +MAX_LINEAR  * magnitude
  backward      linear_x = -MAX_LINEAR  * magnitude
  left          linear_y = +MAX_LINEAR  * magnitude   (strafe left;  +y = left on this robot)
  right         linear_y = -MAX_LINEAR  * magnitude   (strafe right; -y = right on this robot)
  rotate_left   angular_z = +MAX_ANGULAR * magnitude  (CCW; +angular_z = CCW on this robot)
  rotate_right  angular_z = -MAX_ANGULAR * magnitude  (CW;  -angular_z = CW on this robot)
  stop          all zero

Two sub-commands:
  single  — grab one frame, query VLM once, print result + timing
            add --execute to also publish the command once to cmd_vel
  loop    — grab frames continuously, query VLM at --hz, publish cmd_vel
            at PUBLISH_HZ (10 Hz) to keep the firmware watchdog happy

Usage (inside the llm_controller Docker container):
  python3 /opt/robot/llm_controller/my_vla.py single "What should I do?"
  python3 /opt/robot/llm_controller/my_vla.py single --execute "Move to the door"
  python3 /opt/robot/llm_controller/my_vla.py loop "Avoid obstacles and move forward"
  python3 /opt/robot/llm_controller/my_vla.py loop --hz 1 --dry-run "Explore the room"

Environment variables (all optional):
  VLLM_BASE_URL   vLLM API base URL  (default: http://localhost:8000/v1)
  VLM_MODEL       Model to use       (default: Qwen/Qwen3.5-0.8B)
  ROBOT_NS        ROS 2 namespace    (default: hostname with - replaced by _)
"""

import argparse
import base64
import json
import os
import re
import signal
import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from openai import OpenAI
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

# ── Configuration ─────────────────────────────────────────────────────────────

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLM_MODEL     = os.environ.get("VLM_MODEL",     "Qwen/Qwen3.5-0.8B")
ROBOT_NS      = os.environ.get("ROBOT_NS",      socket.gethostname().replace("-", "_"))

DEFAULT_SIZE   = "336"    # VLM input size (px); 336x336 is standard Qwen-VL / LLaVA-1.5
MAX_LINEAR     = 0.3      # m/s  — velocity cap
MAX_ANGULAR    = 0.1     # rad/s — angular rate cap
PUBLISH_HZ     = 10       # cmd_vel publish rate; must satisfy the firmware watchdog
DEFAULT_VLA_HZ = 0.5      # target VLM inference rate (actual is limited by latency)

OUTPUT_DIR  = Path("/opt/robot")
IMAGE_PATH  = OUTPUT_DIR / "vla_frame.jpg"

# ── Action vocabulary ─────────────────────────────────────────────────────────
#
# Each entry: (linear_x_sign, linear_y_sign, angular_z_sign)
# All values are in [-1, +1] and are multiplied by MAX_* and magnitude at
# runtime, so the VLM never needs to know the physical units.

ACTIONS: dict[str, tuple[float, float, float]] = {
    "forward":      ( 1.0,  0.0,  0.0),
    "backward":     (-1.0,  0.0,  0.0),
    "left":         ( 0.0,  1.0,  0.0),   # strafe left  (+linear_y = left on this robot)
    "right":        ( 0.0, -1.0,  0.0),   # strafe right (-linear_y = right on this robot)
    "rotate_left":  ( 0.0,  0.0,  1.0),   # CCW          (+angular_z = CCW on this robot)
    "rotate_right": ( 0.0,  0.0, -1.0),   # CW           (-angular_z = CW on this robot)
    "stop":         ( 0.0,  0.0,  0.0),
}

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""\
You are the vision controller of a ground robot with a wide-angle (fisheye) camera.

Choose ONE action from this list:
  forward       move straight ahead (use only when target is centred)
  backward      move in reverse
  left          strafe left (sideways, no rotation)
  right         strafe right (sideways, no rotation)
  rotate_left   rotate counter-clockwise in place (target is to your left)
  rotate_right  rotate clockwise in place (target is to your right)
  stop          remain stationary

Output ONLY a single JSON object on one line, with no other text:
  "action"    — one of the action names above (required)
  "magnitude" — movement intensity, 0.0 (minimal) to 1.0 (maximum), default 1.0
  "reason"    — one short phrase explaining the choice, mention image position (helps with debugging)

Examples:
  {{"action": "rotate_right", "magnitude": 0.6, "reason": "my target is to the right"}}
  {{"action": "forward",      "magnitude": 0.8, "reason": "my target is ahead of me"}}
  {{"action": "rotate_left",  "magnitude": 0.5, "reason": "my target is to the left"}}
  {{"action": "stop",         "magnitude": 1.0, "reason": "my target is not in the field of view"}}"""

DEFAULT_TASK = (
    "Explore the environment safely. Avoid obstacles. "
    "Prefer moving forward when the path ahead is clear."
)

# ── Image helpers ─────────────────────────────────────────────────────────────

def parse_size(s: str) -> tuple[int, int] | None:
    """Parse "336" → (336,336), "640x480" → (640,480), "native" → None."""
    s = s.strip().lower()
    if s == "native":
        return None
    if "x" in s:
        w, h = s.split("x", 1)
        return int(w), int(h)
    n = int(s)
    return n, n


def jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Return (width, height) by parsing JPEG SOF markers — no full decode needed."""
    i = 0
    while i < len(data) - 1:
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        i += 2
        if marker == 0xD8:
            continue
        if marker in (0xD9, 0xDA):
            break
        length = struct.unpack(">H", data[i : i + 2])[0]
        if marker in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack(">HH", data[i + 3 : i + 7])
            return w, h
        i += length
    raise ValueError("Could not find SOF marker — is this a valid JPEG?")


def resize_jpeg(jpeg_bytes: bytes, width: int, height: int, quality: int = 90) -> bytes:
    """Decode, resize (INTER_AREA), re-encode to JPEG bytes."""
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed — invalid JPEG?")
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return encoded.tobytes()

# ── Action helpers ────────────────────────────────────────────────────────────

def action_to_twist(action: str, magnitude: float) -> Twist:
    """Map a semantic action + magnitude ∈ [0,1] to a Twist message.

    Velocity values are clamped to ±MAX_LINEAR / ±MAX_ANGULAR even if
    magnitude is out of range, so this function is always safe to publish.
    """
    mag = max(0.0, min(1.0, magnitude))
    lx, ly, az = ACTIONS.get(action, (0.0, 0.0, 0.0))
    msg = Twist()
    msg.linear.x  = max(-MAX_LINEAR,  min(MAX_LINEAR,  lx * MAX_LINEAR  * mag))
    msg.linear.y  = max(-MAX_LINEAR,  min(MAX_LINEAR,  ly * MAX_LINEAR  * mag))
    msg.angular.z = max(-MAX_ANGULAR, min(MAX_ANGULAR, az * MAX_ANGULAR * mag))
    return msg


def parse_action(text: str) -> tuple[str, float, str]:
    """
    Extract (action, magnitude, reason) from raw VLM output.
    Returns ("stop", 0.0, "parse error") on failure.
    """
    try:
        m = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            action    = str(data.get("action", "stop")).strip().lower()
            magnitude = float(data.get("magnitude", 1.0))
            reason    = str(data.get("reason", ""))
            if action not in ACTIONS:
                print(f"[warn] Unknown action '{action}', defaulting to stop")
                action = "stop"
            return action, magnitude, reason
    except Exception as e:
        print(f"[warn] Could not parse VLM output '{text.strip()[:80]}': {e}")
    return "stop", 0.0, "parse error"

# ── VLM query ─────────────────────────────────────────────────────────────────

def query_vlm(jpeg_bytes: bytes, task: str) -> tuple[str, dict[str, float]]:
    """
    Send a camera frame + task description to the VLM via streaming.

    Returns:
        raw_text  — full response string
        timing    — dict: ttft, generation, total (seconds), tok/s, tokens
    """
    client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")
    b64      = base64.b64encode(jpeg_bytes).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"

    chunks: list[str] = []
    t_req   = time.perf_counter()
    t_first: float | None = None
    t_last  = t_req
    n_tokens = 0

    stream = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text",      "text": task},
                ],
            },
        ],
        max_tokens=128,
        temperature=0.1,
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        now = time.perf_counter()
        if chunk.usage is not None:
            n_tokens = chunk.usage.completion_tokens
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta is None:
            continue
        if t_first is None:
            t_first = now
        t_last = now
        chunks.append(delta)

    t_first    = t_first or t_last
    ttft       = t_first - t_req
    generation = t_last  - t_first
    total      = t_last  - t_req
    toks_per_s = n_tokens / generation if generation > 0 else 0.0

    timing = {
        "ttft": ttft, "generation": generation,
        "total": total, "tok/s": toks_per_s, "tokens": float(n_tokens),
    }
    return "".join(chunks), timing

# ── One-shot frame grabber (used by single mode) ──────────────────────────────

class _OneShotGrabber(Node):
    """Subscribes to a compressed image topic and captures exactly one frame."""

    def __init__(self, topic: str):
        super().__init__("vla_frame_grabber")
        self.jpeg_bytes: bytes | None = None
        self._sub = self.create_subscription(
            CompressedImage, topic, self._cb, qos_profile_sensor_data
        )
        self.get_logger().info(f"Waiting for a frame on {topic} ...")

    def _cb(self, msg: CompressedImage):
        if self.jpeg_bytes is None:
            self.jpeg_bytes = bytes(msg.data)
            raise SystemExit


def grab_one_frame(topic: str) -> bytes:
    """Block until one JPEG frame arrives on *topic*, then return raw bytes."""
    rclpy.init()
    node = _OneShotGrabber(topic)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    jpeg = node.jpeg_bytes
    node.destroy_node()
    rclpy.shutdown()
    if jpeg is None:
        raise RuntimeError("No frame received before node shut down")
    return jpeg

# ── VLA Controller (loop mode) ────────────────────────────────────────────────

class VLAController(Node):
    """
    ROS 2 node that runs the full VLA loop.

    Three concurrent activities:
      camera subscriber  — stores the latest JPEG in self._latest_jpeg
      infer_thread       — picks up the latest frame, queries the VLM,
                           updates self._current_cmd, paces itself to vla_hz
      publish timer      — pushes self._current_cmd to cmd_vel at PUBLISH_HZ
                           (independent of inference so the watchdog is always fed)
    """

    def __init__(
        self,
        task: str,
        vla_hz: float,
        target_size: tuple[int, int] | None,
        topic: str,
        dry_run: bool,
    ):
        super().__init__("vla_controller")

        self._task        = task
        self._vla_period  = 1.0 / vla_hz
        self._target_size = target_size
        self._dry_run     = dry_run

        self._latest_jpeg: bytes | None = None
        self._jpeg_lock   = threading.Lock()

        self._current_cmd = Twist()
        self._cmd_lock    = threading.Lock()
        self._running     = True

        self._step = 0

        # Camera subscriber
        self._cam_sub = self.create_subscription(
            CompressedImage, topic, self._cam_cb, qos_profile_sensor_data
        )

        # cmd_vel publisher + 10 Hz timer
        cmd_topic = f"/{ROBOT_NS}/cmd_vel"
        self._pub = self.create_publisher(Twist, cmd_topic, 10)
        self.create_timer(1.0 / PUBLISH_HZ, self._publish_cb)

        # Inference thread
        self._infer_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="vla_infer"
        )
        self._infer_thread.start()

        self.get_logger().info("VLA controller ready")
        self.get_logger().info(f"  namespace  : {ROBOT_NS}")
        self.get_logger().info(f"  model      : {VLM_MODEL}")
        self.get_logger().info(f"  camera     : {topic}")
        self.get_logger().info(f"  cmd_vel    : {cmd_topic}")
        self.get_logger().info(f"  vLLM API   : {VLLM_BASE_URL}")
        self.get_logger().info(
            f"  VLA target : {vla_hz:.2f} Hz  "
            f"(actual rate limited by inference latency)"
        )
        self.get_logger().info(f"  publish    : {PUBLISH_HZ} Hz  (firmware watchdog)")
        self.get_logger().info(f"  task       : {task}")
        if dry_run:
            self.get_logger().warn("  DRY RUN — actions logged but NOT published")

    # ── Callbacks / threads ──────────────────────────────────────────────────

    def _cam_cb(self, msg: CompressedImage):
        with self._jpeg_lock:
            self._latest_jpeg = bytes(msg.data)

    def _publish_cb(self):
        if self._dry_run:
            return
        with self._cmd_lock:
            cmd = self._current_cmd
        self._pub.publish(cmd)

    def _inference_loop(self):
        wait = threading.Event()
        while self._running:
            t_iter = time.perf_counter()

            # Wait for first camera frame
            with self._jpeg_lock:
                jpeg = self._latest_jpeg
            if jpeg is None:
                self.get_logger().info("Waiting for camera frame ...")
                wait.wait(timeout=0.5)
                continue

            self._step += 1
            self.get_logger().info(
                f"── VLA step {self._step} "
                + "─" * max(0, 44 - len(str(self._step)))
            )

            # 1. Resize
            t0 = time.perf_counter()
            if self._target_size is not None:
                tw, th = self._target_size
                try:
                    jpeg = resize_jpeg(jpeg, tw, th)
                except Exception as e:
                    self.get_logger().error(f"Resize failed: {e}")
                    wait.wait(timeout=self._vla_period)
                    continue
            t_resize = time.perf_counter() - t0

            # Save frame for external inspection
            try:
                IMAGE_PATH.write_bytes(jpeg)
            except Exception:
                pass

            # 2. VLM query
            t0 = time.perf_counter()
            try:
                raw_text, vlm_t = query_vlm(jpeg, self._task)
            except Exception as e:
                self.get_logger().error(f"VLM query failed: {e}")
                with self._cmd_lock:
                    self._current_cmd = Twist()  # stop on inference error
                wait.wait(timeout=self._vla_period)
                continue
            t_vlm = time.perf_counter() - t0

            # 3. Parse → Twist
            action, magnitude, reason = parse_action(raw_text)
            cmd = action_to_twist(action, magnitude)

            with self._cmd_lock:
                self._current_cmd = cmd

            # 4. Log
            t_step = time.perf_counter() - t_iter
            size_tag = (
                f"{self._target_size[0]}x{self._target_size[1]}"
                if self._target_size else "native"
            )
            self.get_logger().info(
                f"  action : {action:12s}  mag={magnitude:.2f}  reason='{reason}'"
            )
            self.get_logger().info(
                f"  twist  : lx={cmd.linear.x:+.3f}  "
                f"ly={cmd.linear.y:+.3f}  az={cmd.angular.z:+.4f}"
            )
            self.get_logger().info(
                f"  timing : resize={t_resize*1000:.0f}ms  "
                f"ttft={vlm_t['ttft']:.2f}s  "
                f"gen={vlm_t['generation']:.2f}s  "
                f"({int(vlm_t['tokens'])} tok @ {vlm_t['tok/s']:.1f} tok/s)  "
                f"step={t_step:.2f}s  size={size_tag}"
            )

            # 5. Pace: sleep for the remainder of the target period
            elapsed   = time.perf_counter() - t_iter
            remaining = self._vla_period - elapsed
            if remaining > 0:
                wait.wait(timeout=remaining)

# ── CLI helpers ───────────────────────────────────────────────────────────────

def _add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "task",
        nargs="?",
        default=DEFAULT_TASK,
        help="Task description sent to the VLM alongside the camera frame",
    )
    parser.add_argument(
        "--size",
        default=DEFAULT_SIZE,
        metavar="WxH|N|native",
        help=(
            f"Resize the frame before sending to the VLM. "
            f"'N' → NxN, 'WxH' → explicit, 'native' → no resize. "
            f"Default: {DEFAULT_SIZE} (336×336)"
        ),
    )
    parser.add_argument(
        "--topic",
        default=f"/{ROBOT_NS}/camera_0/image_raw/compressed",
        help="ROS 2 compressed image topic to subscribe to",
    )

# ── Sub-command: single ───────────────────────────────────────────────────────

def cmd_single(args) -> int:
    """Grab one frame, query VLM once, print result and timing."""
    try:
        target_size = parse_size(args.size)
    except ValueError as e:
        print(f"[error] Invalid --size: {e}")
        return 1

    print(f"[config] model    : {VLM_MODEL}")
    print(f"[config] vLLM API : {VLLM_BASE_URL}")
    print(f"[config] topic    : {args.topic}")
    print(f"[config] size     : {'native' if target_size is None else f'{target_size[0]}x{target_size[1]}'}")
    print(f"[config] task     : {args.task}")
    print()

    timings: dict[str, float] = {}
    t_wall = time.perf_counter()

    # ── 1. Grab frame ────────────────────────────────────────────────────────
    print("[step 1] Grabbing camera frame ...")
    t0 = time.perf_counter()
    try:
        jpeg = grab_one_frame(args.topic)
    except Exception as e:
        print(f"[error] {e}")
        return 1
    timings["1. frame grab"] = time.perf_counter() - t0
    try:
        w, h = jpeg_dimensions(jpeg)
        size_info = f"{w}x{h} px, {len(jpeg)/1024:.1f} KB"
    except ValueError:
        size_info = f"{len(jpeg)/1024:.1f} KB"
    print(f"         {size_info}  ({timings['1. frame grab']:.2f}s)")

    # ── 2. Resize ────────────────────────────────────────────────────────────
    if target_size is not None:
        tw, th = target_size
        print(f"[step 2] Resizing → {tw}x{th} ...")
        t0 = time.perf_counter()
        try:
            jpeg = resize_jpeg(jpeg, tw, th)
        except Exception as e:
            print(f"[error] Resize failed: {e}")
            return 1
        timings["2. resize"] = time.perf_counter() - t0
        print(f"         {len(jpeg)/1024:.1f} KB  ({timings['2. resize']*1000:.1f}ms)")
    else:
        print(f"[step 2] No resize (native)  {size_info}")

    # ── 3. Save frame ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    IMAGE_PATH.write_bytes(jpeg)
    timings["3. save image"] = time.perf_counter() - t0
    print(f"[step 3] Frame saved to {IMAGE_PATH}  ({timings['3. save image']*1000:.1f}ms)")

    # ── 4. Query VLM ─────────────────────────────────────────────────────────
    print("[step 4] Querying VLM ...")
    print("─" * 60)
    t0 = time.perf_counter()
    try:
        raw_text, vlm_t = query_vlm(jpeg, args.task)
    except Exception as e:
        print(f"[error] VLM query failed: {e}")
        return 1
    timings["4. VLM inference"] = time.perf_counter() - t0
    print(raw_text)
    print("─" * 60)
    print(f"         TTFT:       {vlm_t['ttft']:.2f}s")
    print(f"         Generation: {vlm_t['generation']:.2f}s  "
          f"({int(vlm_t['tokens'])} tokens @ {vlm_t['tok/s']:.1f} tok/s)")

    # ── 5. Parse action ───────────────────────────────────────────────────────
    action, magnitude, reason = parse_action(raw_text)
    cmd = action_to_twist(action, magnitude)
    print()
    print(f"[step 5] Action  : {action}  magnitude={magnitude:.2f}  reason='{reason}'")
    print(f"         Twist   : lx={cmd.linear.x:+.3f}  "
          f"ly={cmd.linear.y:+.3f}  az={cmd.angular.z:+.4f}")

    # ── 6. (Optional) Publish once ────────────────────────────────────────────
    if args.execute:
        print("[step 6] Publishing command once ...")
        rclpy.init()
        node = rclpy.create_node("vla_single_pub")
        pub = node.create_publisher(Twist, f"/{ROBOT_NS}/cmd_vel", 10)
        pub.publish(cmd)
        time.sleep(0.2)   # brief spin so DDS delivers the message
        node.destroy_node()
        rclpy.shutdown()
        print(f"         Published to /{ROBOT_NS}/cmd_vel")
    else:
        print("[step 6] Skipped (pass --execute to publish the command)")

    # ── Timing summary ────────────────────────────────────────────────────────
    total = time.perf_counter() - t_wall
    print()
    print("── Timing summary ──────────────────────────────────────")
    for label, dt in timings.items():
        bar = "█" * int(dt / total * 30)
        print(f"  {label:<22} {dt:6.2f}s  {bar}")
    print(f"  {'TOTAL':<22} {total:6.2f}s")
    print("── VLM breakdown ───────────────────────────────────────")
    print(f"  {'TTFT':<22} {vlm_t['ttft']:6.2f}s")
    print(f"  {'Generation':<22} {vlm_t['generation']:6.2f}s  "
          f"({int(vlm_t['tokens'])} tokens @ {vlm_t['tok/s']:.1f} tok/s)")
    print("────────────────────────────────────────────────────────")
    return 0

# ── Sub-command: loop ─────────────────────────────────────────────────────────

def cmd_loop(args) -> int:
    """Continuous VLA control loop."""
    try:
        target_size = parse_size(args.size)
    except ValueError as e:
        print(f"[error] Invalid --size: {e}")
        return 1

    rclpy.init()
    node = VLAController(
        task=args.task,
        vla_hz=args.hz,
        target_size=target_size,
        topic=args.topic,
        dry_run=args.dry_run,
    )

    def _sigint(sig, frame):
        node._running = False
        if not args.dry_run:
            node._pub.publish(Twist())   # publish zero velocity before shutting down
        rclpy.shutdown()

    signal.signal(signal.SIGINT, _sigint)

    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        node.destroy_node()
    return 0

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="my_vla.py",
        description="VLA controller: VLM-driven robot control from live camera images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
action vocabulary:
  forward, backward, left, right, rotate_left, rotate_right, stop

examples:
  # Single query — print action, no motion
  python3 my_vla.py single "What should I do to reach the open door?"

  # Single query + publish the resulting command once
  python3 my_vla.py single --execute "Move toward the brightest area"

  # Continuous VLA loop at default 0.5 Hz
  python3 my_vla.py loop "Avoid obstacles and move forward"

  # 1 Hz loop, observe actions without moving
  python3 my_vla.py loop --hz 1 --dry-run "Explore the room"

  # Use a different model / endpoint
  VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct python3 my_vla.py loop "Follow the corridor"
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── single ───────────────────────────────────────────────────────────────
    p_single = sub.add_parser(
        "single",
        help="Grab one frame, query VLM once, print result and timing",
    )
    _add_common_args(p_single)
    p_single.add_argument(
        "--execute",
        action="store_true",
        help="Also publish the resulting Twist to cmd_vel once (default: dry-run)",
    )

    # ── loop ─────────────────────────────────────────────────────────────────
    p_loop = sub.add_parser(
        "loop",
        help=f"Continuous VLA control loop (cmd_vel at {PUBLISH_HZ} Hz, VLM at --hz)",
    )
    _add_common_args(p_loop)
    p_loop.add_argument(
        "--hz",
        type=float,
        default=DEFAULT_VLA_HZ,
        metavar="RATE",
        help=(
            f"Target VLM inference rate in Hz (default: {DEFAULT_VLA_HZ}). "
            "Actual rate is limited by inference latency; the publish timer "
            f"always runs at {PUBLISH_HZ} Hz regardless."
        ),
    )
    p_loop.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions and timing but do NOT publish to cmd_vel",
    )

    args = parser.parse_args()
    raise SystemExit(
        cmd_single(args) if args.command == "single" else cmd_loop(args)
    )


if __name__ == "__main__":
    main()
e