"""Bootstrap the Robot V2 environment server inside Blender."""

from __future__ import annotations

import runpy
from pathlib import Path

import bpy


ENV_CONTROL_PATH = Path(__file__).resolve().with_name("EnvControl.py")
runpy.run_path(str(ENV_CONTROL_PATH), run_name="__main__")

result = bpy.ops.wm.rl_env_server_modal()
print(f"bpy.ops.wm.rl_env_server_modal() returned: {result}")
