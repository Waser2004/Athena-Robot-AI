"""Client interface for controlling the Blender robot virtual environment."""

from __future__ import annotations

import json
import socket
import struct
from typing import Any, Sequence


class EnvInterfaceError(RuntimeError):
    """Raised when communication with the robot virtual environment fails."""


class EnvInteface:
    """
    TCP client for EnvControl's Blender server.

    Naming keeps the requested `EnvInteface` class spelling.
    """

    def __init__(self, host: str = "localhost", port: int = 5055, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket: socket.socket | None = None

    def connect(self) -> None:
        """Open a TCP connection to the running Blender environment server."""
        if self._socket is not None:
            return
        try:
            self._socket = socket.create_connection((self.host, self.port), timeout=self.timeout)
            self._socket.settimeout(self.timeout)
        except OSError as exc:
            self._socket = None
            raise EnvInterfaceError(
                f"Could not connect to robot env server at {self.host}:{self.port}"
            ) from exc

    def close(self) -> None:
        """Close the TCP connection if it is open."""
        if self._socket is None:
            return
        try:
            self._socket.close()
        finally:
            self._socket = None

    def __enter__(self) -> "EnvInteface":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def is_connected(self) -> bool:
        return self._socket is not None

    def reset(self, cube_position: str = "home", robot_pose: str = "home") -> None:
        """Reset cube position and robot pose in the environment."""
        self._send_request(
            function="reset",
            args={"cube_position": cube_position, "robot_pose": robot_pose},
            expect_response=False,
        )

    def get_state(
        self,
        actuator_rotations: bool = True,
        actuator_velocities: bool = True,
        target_cube_state: bool = True,
        graper: bool = True,
        collisions: bool = True,
        workplate_coverage: bool = True,
        distance_to_target: bool = True,
        image: bool = False,
    ) -> dict[str, Any]:
        """Fetch selected state values from the environment."""
        result = self._send_request(
            function="get_state",
            args={
                "actuator_rotations": actuator_rotations,
                "actuator_velocities": actuator_velocities,
                "target_cube_state": target_cube_state,
                "graper": graper,
                "collisions": collisions,
                "workplate_coverage": workplate_coverage,
                "distance_to_target": distance_to_target,
                "image": image,
            },
            expect_response=True,
        )
        if not isinstance(result, dict):
            raise EnvInterfaceError("Unexpected response type for get_state")
        return result

    def step(self, actuator_velocities: Sequence[float], grapper_state: bool) -> float:
        """Apply one action step and return the resulting motion cost."""
        velocities = self._ensure_six_values(actuator_velocities, "actuator_velocities")
        result = self._send_request(
            function="step",
            args={"actuator_velocities": velocities, "grapper_state": bool(grapper_state)},
            expect_response=True,
        )
        return float(result)

    def set_robot_pose(self, actuator_rotations: Sequence[float]) -> None:
        """Set robot joints directly in radians."""
        rotations = self._ensure_six_values(actuator_rotations, "actuator_rotations")
        self._send_request(
            function="set_robot_pose",
            args={"actuator_rotations": rotations},
            expect_response=False,
        )

    def set_cube_pose(
        self,
        x: float,
        y: float,
        z: float = 0.025,
        yaw: float | None = None,
    ) -> None:
        """Set target cube position and optional yaw (radians)."""
        args: dict[str, Any] = {"x": float(x), "y": float(y), "z": float(z)}
        if yaw is not None:
            args["yaw"] = float(yaw)
        self._send_request(function="set_cube_pose", args=args, expect_response=False)

    def target_cube_in_view(self) -> float:
        """Return normalized distance of cube center from image center."""
        result = self._send_request(
            function="target_cube_in_view",
            args={},
            expect_response=True,
        )
        return float(result)

    def target_cube_within_padding(self, padding: float = 0.1) -> bool:
        """Return True if projected cube bbox lies inside camera view with given padding."""
        result = self._send_request(
            function="target_cube_within_padding",
            args={"padding": float(padding)},
            expect_response=True,
        )
        return bool(result)

    def cube_visibility_labels(self) -> dict[str, Any]:
        """Return cube visibility labels used by the cube detection data generator."""
        result = self._send_request(
            function="cube_visibility_labels",
            args={},
            expect_response=True,
        )
        if not isinstance(result, dict):
            raise EnvInterfaceError("Unexpected response type for cube_visibility_labels")
        return result

    def call(self, function: str, args: dict[str, Any] | None = None, expect_response: bool = True) -> Any:
        """Low-level access for additional server functions."""
        return self._send_request(function=function, args=args or {}, expect_response=expect_response)

    def _send_request(self, function: str, args: dict[str, Any], expect_response: bool) -> Any:
        self._ensure_connected()
        request = {"function": function, "args": args}
        payload = json.dumps(request).encode("utf-8")
        packet = struct.pack(">I", len(payload)) + payload

        try:
            self._socket.sendall(packet)
            if not expect_response:
                return None
            return self._recv_result()
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.close()
            raise EnvInterfaceError(f"Request '{function}' failed") from exc

    def _recv_result(self) -> Any:
        header = self._recv_exact(4)
        msg_len = struct.unpack(">I", header)[0]
        body = self._recv_exact(msg_len)
        response = json.loads(body.decode("utf-8"))
        if "result" not in response:
            raise EnvInterfaceError("Response missing 'result' field")
        return response["result"]

    def _recv_exact(self, size: int) -> bytes:
        self._ensure_connected()
        data = bytearray()
        while len(data) < size:
            chunk = self._socket.recv(size - len(data))
            if not chunk:
                raise EnvInterfaceError("Connection closed by server")
            data.extend(chunk)
        return bytes(data)

    def _ensure_connected(self) -> None:
        if self._socket is None:
            self.connect()

    @staticmethod
    def _ensure_six_values(values: Sequence[float], arg_name: str) -> list[float]:
        output = [float(value) for value in values]
        if len(output) != 6:
            raise ValueError(f"{arg_name} must have exactly 6 values")
        return output


class EnvInterface(EnvInteface):
    """Compatibility alias with corrected spelling."""


__all__ = ["EnvInteface", "EnvInterface", "EnvInterfaceError"]
