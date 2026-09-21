"""Shared viewer arguments and passive loop for environment demos."""

import time

import numpy as np


def add_viewer_arguments(parser, passive=True):
    parser.add_argument("--renderer", choices=["mjviser", "mjviewer"], default="mjviser")
    parser.add_argument("--viewer-host", default="127.0.0.1")
    parser.add_argument("--viewer-port", type=int, default=8080)
    if passive:
        parser.add_argument("--steps", type=int, default=None,
                            help="Stop passive viewing after this many simulation steps")


def renderer_config(args):
    if args.renderer == "mjviser":
        return {"host": args.viewer_host, "port": args.viewer_port}
    return None


def run_passive_viewer(env, steps=None):
    """Display a settled environment without importing a teleoperation device."""
    if steps is not None and steps < 0:
        raise ValueError("steps must be nonnegative")
    try:
        env.reset()
        env.render()
        print("Passive viewing (no teleop). Press Ctrl+C to exit.")
        action = np.zeros(env.action_dim)
        count = 0
        while steps is None or count < steps:
            start = time.monotonic()
            env.step(action)  # The renderer updates automatically after stepping.
            count += 1
            time.sleep(max(0, 1 / env.control_freq - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
