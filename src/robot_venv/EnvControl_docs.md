# EnvControl Interface Reference

This document describes the interface exposed by `EnvControl.py` for controlling the Blender robot environment from external scripts.

## Overview

`EnvControl.py` exposes a TCP server inside Blender through the modal operator:

- Blender operator: `wm.rl_env_server_modal`
- Host: `localhost`
- Port: `5055`
- Transport: TCP
- Message framing: 4-byte big-endian unsigned length header + UTF-8 JSON payload

The server creates one `RobotEnv` instance (`fps=30`) and handles function-style JSON RPC requests.

## Direct Blender Python API (`RobotEnv`)

If your script runs inside Blender (instead of TCP), you can use the `RobotEnv` class directly.

### Constructor

- `RobotEnv(fps: int = 30)`

### Public Methods

- `set_robot_pose(actuator_rotations: list[float]) -> None`
- `reset(cube_position: str = "home", robot_pose: str = "home", actuator_rotations: list[float] | None = None) -> None`
- `target_cube_in_view() -> float`
- `get_state(actuator_rotations=True, actuator_velocities=True, target_cube_state=True, graper=True, collisions=True, workplate_coverage=True, distance_to_target=True, image=False) -> dict`
- `step(actuator_velocities: list[float] | None = None, grapper_state: bool | None = None) -> float`

Notes:

- Rotations are radians, velocity inputs are treated as degrees/second.
- `reset(..., actuator_rotations=...)` currently exposes the parameter but does not use it in implementation.

## Starting The Server In Blender

1. Open the `.blend` scene that contains all required objects.
2. Run `EnvControl.py` in Blender's scripting workspace.
3. Press `F3` and run: `Start RL Env Server (Modal Timer)`.

When active, Blender prints: `RL Env Server started (modal timer)`.

## Wire Protocol

Every request must be encoded as:

1. `json.dumps(request).encode("utf-8")`
2. Prefix with 4-byte big-endian payload length (`struct.pack('>I', len(payload))`)
3. Send `header + payload`

Responses (for functions that return values) use the same framing with payload:

```json
{"result": ...}
```

## Request Format

All requests have the same top-level shape:

```json
{
  "function": "<function_name>",
  "args": { ... }
}
```

Supported `function` values:

- `reset`
- `get_state`
- `step`
- `target_cube_in_view`
- `set_robot_pose`

## Function Reference

### 1. `reset`

Reset environment state (cube pose, robot pose, random light settings).

Request:

```json
{
  "function": "reset",
  "args": {
    "cube_position": "home",
    "robot_pose": "home"
  }
}
```

Arguments:

- `cube_position` (`str`): one of
  - `"home"`
  - `"random_on_workplate"`
  - `"random_not_on_workplate"` (currently placeholder)
- `robot_pose` (`str`): one of
  - `"home"`
  - `"resting"`
  - `"random"` (currently placeholder)

Response:

- No response payload is sent for `reset`.

Notes:

- Invalid values cause assertion errors in Blender.
- Lighting energy and color are randomized on every reset.

### 2. `get_state`

Return selected parts of the environment state.

Request:

```json
{
  "function": "get_state",
  "args": {
    "actuator_rotations": true,
    "actuator_velocities": true,
    "target_cube_state": true,
    "graper": true,
    "collisions": true,
    "workplate_coverage": true,
    "distance_to_target": true,
    "image": false
  }
}
```

Arguments (all `bool`):

- `actuator_rotations`
- `actuator_velocities`
- `target_cube_state`
- `graper` (spelling in API is exactly `graper`)
- `collisions`
- `workplate_coverage`
- `distance_to_target`
- `image`

Response:

```json
{
  "result": {
    "actuator_rotations": [r0, r1, r2, r3, r4, r5],
    "actuator_velocities": [v0, v1, v2, v3, v4, v5],
    "target_cube_location": [x, y, z],
    "target_cube_rotation": [rx, ry, rz],
    "graper": false,
    "collisions": false,
    "workplate_coverage": [true, false, ...],
    "distance_to_target": 0.123,
    "relative_rotation": [roll, pitch, yaw],
    "image": [[...], [...], ...]
  }
}
```

Returned keys depend on which flags were set to `true`.

Units and value meanings:

- Rotations are in radians.
- `actuator_velocities` are in degrees/second as currently stored in env action state.
- `distance_to_target` is Euclidean world-space distance in Blender units.
- `workplate_coverage` is a boolean list indicating visibility of each predefined grid point from camera view.
- `image` is grayscale pixel data as nested Python lists (`height x width`), values from Blender compositor output.

### 3. `step`

Apply one control step to robot joints and gripper, then return motion cost.

Request:

```json
{
  "function": "step",
  "args": {
    "actuator_velocities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "grapper_state": false
  }
}
```

Arguments:

- `actuator_velocities` (`list[float]` length 6): joint velocity commands.
- `grapper_state` (`bool`): `true` closes gripper, `false` opens gripper.

Response:

```json
{"result": <cost_float>}
```

Cost model (current implementation):

- Sum of normalized motor effort terms from all 6 joints.
- Add `0.1` when gripper toggles state.

### 4. `target_cube_in_view`

Compute normalized distance of cube center from image center using visible cube vertices.

Request:

```json
{
  "function": "target_cube_in_view",
  "args": {}
}
```

Response:

```json
{"result": <float_in_0_to_1>}
```

Interpretation:

- `0.0`: cube centered in image.
- `1.0`: no cube vertex visible (or maximal normalized offset).

### 5. `set_robot_pose`

Set exact robot joint rotations directly.

Request:

```json
{
  "function": "set_robot_pose",
  "args": {
    "actuator_rotations": [r0, r1, r2, r3, r4, r5]
  }
}
```

Arguments:

- `actuator_rotations` (`list[float]` length 6): target joint rotations in radians.

Response:

- No response payload is sent for `set_robot_pose`.

## Joint Mapping

The 6 actuator entries map to robot joints in this exact order:

1. base `z`
2. primary arm `x`
3. secondary arm part 1 `x`
4. secondary arm part 2 `z`
5. tertiary arm part 1 `x`
6. tertiary arm part 2 `y`

## Python Client Example

```python
import json
import socket
import struct

HOST = "localhost"
PORT = 5055


def send_request(sock, request, expect_response=True):
    payload = json.dumps(request).encode("utf-8")
    sock.sendall(struct.pack(">I", len(payload)) + payload)

    if not expect_response:
        return None

    header = sock.recv(4)
    if not header:
        raise RuntimeError("No response header")

    msg_len = struct.unpack(">I", header)[0]
    data = b""
    while len(data) < msg_len:
        chunk = sock.recv(msg_len - len(data))
        if not chunk:
            raise RuntimeError("Connection closed while reading response")
        data += chunk

    return json.loads(data.decode("utf-8"))


with socket.create_connection((HOST, PORT)) as sock:
    # reset (no response)
    send_request(sock, {
        "function": "reset",
        "args": {"cube_position": "home", "robot_pose": "home"}
    }, expect_response=False)

    # step (response)
    step_resp = send_request(sock, {
        "function": "step",
        "args": {
            "actuator_velocities": [0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
            "grapper_state": False
        }
    })
    print("step cost:", step_resp["result"])

    # get_state (response)
    state_resp = send_request(sock, {
        "function": "get_state",
        "args": {
            "actuator_rotations": True,
            "actuator_velocities": True,
            "target_cube_state": True,
            "graper": True,
            "collisions": True,
            "workplate_coverage": False,
            "distance_to_target": True,
            "image": False
        }
    })
    print("state keys:", state_resp["result"].keys())
```

## Implementation Notes And Caveats

- `graper`/`grapper` spelling is inconsistent in names but must be used exactly as implemented in request keys.
- `reset` and `set_robot_pose` do not currently send acknowledgment messages.
- `workplate_coverage` depends on `docs/grid_centers.txt` loaded via an absolute Windows path in `EnvControl.py`.
- The server handles one client connection at a time.
- Requests are processed on Blender timer events; this is not a high-frequency real-time loop.
