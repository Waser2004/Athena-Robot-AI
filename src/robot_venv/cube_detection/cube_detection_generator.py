"""Blender-side generator for cube-visibility detection data."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


# Public constants required by the generation workflow.
CAMERA_MIN_Z_DIST_TO_WORKPLATE = 0.10
CAMERA_MAX_Z_DIST_TO_WORKPLATE = 0.35
TRACK_ITER_MOVE_DIST = 0.02


class CubeDetectionGenerator:
    """
    Generate cube-visibility image sequences inside Cube_detection_env.blend.

    Required objects in the active .blend:
    - OV2640
    - OV2640 track
    - Target Cube
    - Workplate
    """

    def __init__(self) -> None:
        self.camera, self.track, self.target_cube, self.workplate = self._bind_required_objects()

    @staticmethod
    def _bind_required_objects():
        required_names = ("OV2640", "OV2640 track", "Target Cube", "Workplate")
        missing = [name for name in required_names if name not in bpy.data.objects]
        if missing:
            raise RuntimeError(
                "Missing required Blender objects: "
                + ", ".join(missing)
            )

        return (
            bpy.data.objects["OV2640"],
            bpy.data.objects["OV2640 track"],
            bpy.data.objects["Target Cube"],
            bpy.data.objects["Workplate"],
        )

    @staticmethod
    def _normalize_cube_pos(cube_pos: Sequence[float] | Vector) -> tuple[float, float, float]:
        if len(cube_pos) == 2:
            return float(cube_pos[0]), float(cube_pos[1]), 0.025
        if len(cube_pos) == 3:
            return float(cube_pos[0]), float(cube_pos[1]), float(cube_pos[2])
        raise ValueError("cube_pos must have length 2 or 3")

    @staticmethod
    def _workplate_x_max_for_y(y_m: float) -> float:
        if y_m <= -0.16:
            return 0.05

        # EnvControl-equivalent upper x-bound for y in (-0.16, -0.10].
        radial = (-(y_m * 1000.0) - 25.0)
        radicand = (135.0**2) - (radial**2)
        if radicand < 0.0:
            return -0.24
        return -(math.sqrt(radicand) + 25.0) / 1000.0

    def _is_valid_workplate_xy(self, x_m: float, y_m: float) -> bool:
        if y_m < -0.615 or y_m > -0.1:
            return False
        if x_m < -0.24:
            return False
        return x_m <= self._workplate_x_max_for_y(y_m)

    def _sample_valid_workplate_xy(self, rng: random.Random) -> tuple[float, float]:
        for _ in range(5000):
            y_m = rng.uniform(-0.615, -0.1)
            x_m = rng.uniform(-0.24, 0.05)
            if self._is_valid_workplate_xy(x_m=x_m, y_m=y_m):
                return x_m, y_m
        raise RuntimeError("Failed to sample a valid workplate position")

    def _workplate_top_z(self) -> float:
        corners_world = [self.workplate.matrix_world @ Vector(corner) for corner in self.workplate.bound_box]
        return max(corner.z for corner in corners_world)

    def _camera_has_track_constraint(self) -> bool:
        for constraint in self.camera.constraints:
            if constraint.type == "TRACK_TO" and constraint.target == self.track:
                return True
        return False

    def _look_at_track_fallback(self) -> None:
        direction = self.track.location - self.camera.location
        if direction.length <= 1e-9:
            return
        self.camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    def _update_view_layer(self) -> None:
        bpy.context.view_layer.update()
        if not self._camera_has_track_constraint():
            self._look_at_track_fallback()
            bpy.context.view_layer.update()

    def reset(self, cube_pos: Sequence[float] | Vector, seed: int | None = None) -> dict[str, Any]:
        """
        Reset scene state for one sample.

        - Set cube to `cube_pos` (x, y, z) or (x, y) with z=0.025.
        - Set OV2640 track exactly to cube position.
        - Randomize camera position above the workplate.
        """
        rng = random.Random(seed)

        cube_x, cube_y, cube_z = self._normalize_cube_pos(cube_pos)
        self.target_cube.location.x = cube_x
        self.target_cube.location.y = cube_y
        self.target_cube.location.z = cube_z

        self.track.location.x = cube_x
        self.track.location.y = cube_y
        self.track.location.z = cube_z

        cam_x, cam_y = self._sample_valid_workplate_xy(rng=rng)
        cam_z = self._workplate_top_z() + rng.uniform(
            CAMERA_MIN_Z_DIST_TO_WORKPLATE,
            CAMERA_MAX_Z_DIST_TO_WORKPLATE,
        )

        self.camera.location.x = cam_x
        self.camera.location.y = cam_y
        self.camera.location.z = cam_z
        self._update_view_layer()

        return {
            "cube_pos": (cube_x, cube_y, cube_z),
            "camera_pos": (float(self.camera.location.x), float(self.camera.location.y), float(self.camera.location.z)),
            "track_pos": (float(self.track.location.x), float(self.track.location.y), float(self.track.location.z)),
            "workplate_top_z": self._workplate_top_z(),
        }

    def _project_cube_vertices_ndc(self) -> list[tuple[float, float, float]]:
        self._update_view_layer()

        if self.target_cube.type != "MESH":
            raise TypeError("Target Cube must be a mesh object")

        scene = bpy.context.scene
        world_matrix = self.target_cube.matrix_world

        projected_vertices: list[tuple[float, float, float]] = []
        for vertex in self.target_cube.data.vertices:
            world_coord = world_matrix @ vertex.co
            co_ndc = world_to_camera_view(scene, self.camera, world_coord)
            projected_vertices.append((float(co_ndc.x), float(co_ndc.y), float(co_ndc.z)))
        return projected_vertices

    def _calculate_visibility_labels(self) -> dict[str, float | bool | str]:
        """
        Calculate continuous cube-visibility labels in normalized image coordinates.

        Metrics:
        - `visible_image_ratio`: clipped cube bbox area inside frame (0..1)
        - `inframe_fraction`: fraction of projected cube bbox that lies inside frame (0..1)
        - `edge_margin`: minimum signed distance of projected bbox to frame borders
                         (>0: fully inside, <0: crossing/outside)
        """
        projected_vertices = self._project_cube_vertices_ndc()
        front_vertices = [(x, y) for x, y, z in projected_vertices if z > 0.0]

        if not front_vertices:
            return {
                "cube_visible": False,
                "label": "no_cube_visible",
                "visible_image_ratio": 0.0,
                "inframe_fraction": 0.0,
                "edge_margin": -1.0,
            }

        xs = [x for x, _ in front_vertices]
        ys = [y for _, y in front_vertices]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        raw_w = max(max_x - min_x, 0.0)
        raw_h = max(max_y - min_y, 0.0)
        raw_area = raw_w * raw_h

        clip_min_x = max(min_x, 0.0)
        clip_max_x = min(max_x, 1.0)
        clip_min_y = max(min_y, 0.0)
        clip_max_y = min(max_y, 1.0)

        clipped_w = max(clip_max_x - clip_min_x, 0.0)
        clipped_h = max(clip_max_y - clip_min_y, 0.0)
        clipped_area = clipped_w * clipped_h

        visible_image_ratio = min(max(clipped_area, 0.0), 1.0)
        if raw_area > 1e-12:
            inframe_fraction = min(max(clipped_area / raw_area, 0.0), 1.0)
        else:
            inframe_fraction = 1.0 if clipped_area > 0.0 else 0.0

        edge_margin = min(min_x, 1.0 - max_x, min_y, 1.0 - max_y)
        cube_visible = clipped_area > 0.0
        if not cube_visible:
            cube_visible = any(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in front_vertices)

        return {
            "cube_visible": bool(cube_visible),
            "label": "cube_visible" if cube_visible else "no_cube_visible",
            "visible_image_ratio": float(visible_image_ratio),
            "inframe_fraction": float(inframe_fraction),
            "edge_margin": float(edge_margin),
        }

    @staticmethod
    def _render_png(output_file: Path) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        scene = bpy.context.scene

        prev_filepath = scene.render.filepath
        prev_format = scene.render.image_settings.file_format

        try:
            scene.render.filepath = str(output_file)
            scene.render.image_settings.file_format = "PNG"
            bpy.ops.render.render(write_still=True)
        finally:
            scene.render.filepath = prev_filepath
            scene.render.image_settings.file_format = prev_format

    @staticmethod
    def _sample_direction_xy(rng: random.Random) -> tuple[float, float]:
        theta = rng.uniform(0.0, 2.0 * math.pi)
        return math.cos(theta), math.sin(theta)

    def _move_track_step_xy(self, dx: float, dy: float) -> None:
        self.track.location.x += dx
        self.track.location.y += dy
        self._update_view_layer()

    def _default_cube_pos(self, rng: random.Random) -> tuple[float, float, float]:
        x_m, y_m = self._sample_valid_workplate_xy(rng=rng)
        return x_m, y_m, 0.025

    @staticmethod
    def _build_frame_filename(
        sample_idx: int,
        frame_idx: int,
        label: str,
        visible_image_ratio: float,
        inframe_fraction: float,
        edge_margin: float,
    ) -> str:
        return (
            f"s_{sample_idx:06d}__f_{frame_idx:04d}__label_{label}"
            f"__vis_{visible_image_ratio:.6f}"
            f"__infrm_{inframe_fraction:.6f}"
            f"__edge_{edge_margin:.6f}.png"
        )

    def generate_samples(
        self,
        n_samples: int,
        output_dir: str | Path,
        cube_positions: Iterable[Sequence[float]] | None = None,
        max_steps_per_sample: int = 20,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate sequences by moving OV2640 track away from the cube until invisible.

        For each sample, all frames are saved up to `max_steps_per_sample`.
        If the cube is still visible after that many frames, generation moves on to
        the next sample without raising an error.
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        if max_steps_per_sample <= 0:
            raise ValueError("max_steps_per_sample must be > 0")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(seed)
        provided_positions = list(cube_positions) if cube_positions is not None else []
        if provided_positions and len(provided_positions) < n_samples:
            raise ValueError("cube_positions must provide at least n_samples entries")

        metadata: list[dict[str, Any]] = []

        for sample_idx in range(n_samples):
            cube_pos = (
                self._normalize_cube_pos(provided_positions[sample_idx])
                if provided_positions
                else self._default_cube_pos(rng=rng)
            )

            self.reset(cube_pos=cube_pos, seed=rng.randint(0, 2_147_483_647))
            dir_x, dir_y = self._sample_direction_xy(rng=rng)

            terminal_frame_written = False
            for frame_idx in range(max_steps_per_sample):
                labels = self._calculate_visibility_labels()
                label = str(labels["label"])
                visible = bool(labels["cube_visible"])

                file_name = self._build_frame_filename(
                    sample_idx=sample_idx,
                    frame_idx=frame_idx,
                    label=label,
                    visible_image_ratio=float(labels["visible_image_ratio"]),
                    inframe_fraction=float(labels["inframe_fraction"]),
                    edge_margin=float(labels["edge_margin"]),
                )
                file_path = out_dir / file_name
                self._render_png(file_path)

                metadata.append(
                    {
                        "sample_index": sample_idx,
                        "frame_index": frame_idx,
                        "label": label,
                        "file_path": str(file_path),
                        "cube_pos": tuple(float(v) for v in cube_pos),
                        "direction_xy": (float(dir_x), float(dir_y)),
                        "visible_image_ratio": float(labels["visible_image_ratio"]),
                        "inframe_fraction": float(labels["inframe_fraction"]),
                        "edge_margin": float(labels["edge_margin"]),
                        "track_pos": (
                            float(self.track.location.x),
                            float(self.track.location.y),
                            float(self.track.location.z),
                        ),
                    }
                )

                if not visible:
                    terminal_frame_written = True
                    break

                if frame_idx < max_steps_per_sample - 1:
                    self._move_track_step_xy(
                        dx=dir_x * TRACK_ITER_MOVE_DIST,
                        dy=dir_y * TRACK_ITER_MOVE_DIST,
                    )

            if not terminal_frame_written:
                print(
                    "Info: reached max_steps_per_sample without terminal no_cube_visible frame "
                    f"for sample {sample_idx}; continuing with next sample."
                )

        return metadata


__all__ = [
    "CAMERA_MIN_Z_DIST_TO_WORKPLATE",
    "CAMERA_MAX_Z_DIST_TO_WORKPLATE",
    "TRACK_ITER_MOVE_DIST",
    "CubeDetectionGenerator",
]
