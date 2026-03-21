"""Scaffolding for grid-based cube localisation data generation."""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from math import pi, radians, sqrt
from pathlib import Path
from typing import Any, Iterable

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from robot_venv.EnvInterface import EnvInteface

try:
    from PIL import Image
except ImportError as exc:
    raise ImportError("Pillow is required for PNG export. Install with: pip install Pillow") from exc


DEFAULT_SEARCH_PATH_FILE = Path(__file__).resolve().parents[2] / "docs" / "search_path.json"
DEFAULT_CUBE_LOCALISATION_DATASET_DIR = Path(__file__).resolve().parents[2] / "docs" / "Cube_Localisation_dataset_tmp"


def load_search_path(path: Path | None = None) -> list[list[float]]:
    """Load optional robot search-path waypoints from JSON."""
    searchpath_file = path or DEFAULT_SEARCH_PATH_FILE
    with searchpath_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        payload = payload.get("search_path", [])

    return [list(map(float, waypoint)) for waypoint in payload]


@dataclass(frozen=True)
class WorkplateBoundsCm:
    min_x_cm: float
    max_x_cm: float
    min_y_cm: float
    max_y_cm: float

    def validate(self) -> None:
        if self.min_x_cm >= self.max_x_cm:
            raise ValueError("min_x_cm must be smaller than max_x_cm")
        if self.min_y_cm >= self.max_y_cm:
            raise ValueError("min_y_cm must be smaller than max_y_cm")


@dataclass(frozen=True)
class DataGenerationConfig:
    box_size_cm: float
    iteration_amount: int
    workplate_bounds_cm: WorkplateBoundsCm = WorkplateBoundsCm(
        min_x_cm=-24.0,
        max_x_cm=5.0,
        min_y_cm=-61.5,
        max_y_cm=-10.0,
    )
    cube_z_cm: float = 2.5
    random_yaw: bool = True
    z_rotation_min_rad: float = -pi / 4
    z_rotation_max_rad: float = pi / 4
    z_rotation_intervals: int = 1
    rotation_iteration_amount: int = 1
    robot_pose: str = "home"
    reset_before_each_sample: bool = True
    include_image: bool = False
    seed: int | None = None

    def validate(self) -> None:
        if self.box_size_cm <= 0:
            raise ValueError("box_size_cm must be > 0")
        if self.iteration_amount <= 0:
            raise ValueError("iteration_amount must be > 0")
        if self.cube_z_cm <= 0:
            raise ValueError("cube_z_cm must be > 0")
        if self.z_rotation_intervals <= 0:
            raise ValueError("z_rotation_intervals must be > 0")
        if self.rotation_iteration_amount <= 0:
            raise ValueError("rotation_iteration_amount must be > 0")
        if self.z_rotation_min_rad > self.z_rotation_max_rad:
            raise ValueError("z_rotation_min_rad must be <= z_rotation_max_rad")
        self.workplate_bounds_cm.validate()


@dataclass(frozen=True)
class GridCell:
    cell_index: int
    row: int
    col: int
    x_min_cm: float
    x_max_cm: float
    y_min_cm: float
    y_max_cm: float


class DataGenerator:
    """Grid-based data generator for cube localisation."""

    ACTUATOR_MAX_SPEEDS_DEG_S = [6.7, 6.7, 6.7, 9.5, 6.7, 9.5]
    CONTROL_FPS = 30.0
    # Mapping from commanded actuator velocity sign to observed joint-angle change sign
    # in EnvControl.step():
    # z0 -= v0, x1 += v1, x2 -= v2, z3 -= v3, x4 += v4, y5 -= v5
    ACTUATOR_STEP_DIRECTIONS = [-1.0, 1.0, -1.0, -1.0, 1.0, -1.0]

    def __init__(self, env: EnvInteface, search_path: list[list[float]] | None = None) -> None:
        self.env = env
        self.search_path = search_path if search_path is not None else load_search_path()

    @staticmethod
    def _build_bins(min_cm: float, max_cm: float, box_size_cm: float) -> list[tuple[float, float]]:
        bins: list[tuple[float, float]] = []
        cursor = min_cm
        while cursor < max_cm:
            upper = min(cursor + box_size_cm, max_cm)
            bins.append((cursor, upper))
            cursor = upper
        return bins

    def build_workplate_grid(self, config: DataGenerationConfig) -> list[GridCell]:
        """Overlay the workplate with `box_size_cm` cells."""
        config.validate()
        bounds = config.workplate_bounds_cm

        x_bins = self._build_bins(bounds.min_x_cm, bounds.max_x_cm, config.box_size_cm)
        y_bins = self._build_bins(bounds.min_y_cm, bounds.max_y_cm, config.box_size_cm)

        cells: list[GridCell] = []
        cell_index = 0
        for row, (y_min, y_max) in enumerate(y_bins):
            for col, (x_min, x_max) in enumerate(x_bins):
                cells.append(
                    GridCell(
                        cell_index=cell_index,
                        row=row,
                        col=col,
                        x_min_cm=x_min,
                        x_max_cm=x_max,
                        y_min_cm=y_min,
                        y_max_cm=y_max,
                    )
                )
                cell_index += 1
        return cells

    @staticmethod
    def _cm_to_m(value_cm: float) -> float:
        return value_cm / 100.0

    @staticmethod
    def _env_max_x_cm_for_y_cm(y_cm: float) -> float:
        """
        Exact bound derived from `EnvControl.py` random_on_workplate logic.
        """
        if y_cm <= -16.0:
            return 5.0

        # Original EnvControl math in mm:
        # lower_mm = sqrt(135**2 - (-y_m*1000 - 25)**2) + 25
        # x_m = -random.uniform(lower_mm, 240) / 1000
        y_mm = -10.0 * y_cm
        radicand = 135.0**2 - (y_mm - 25.0) ** 2
        if radicand < 0.0:
            return -24.0
        lower_mm = sqrt(radicand) + 25.0
        return -lower_mm / 10.0

    def _is_within_env_workplate(self, x_cm: float, y_cm: float) -> bool:
        """Check whether a position is valid per EnvControl workplate bounds."""
        if y_cm < -61.5 or y_cm > -10.0:
            return False
        if x_cm < -24.0:
            return False
        return x_cm <= self._env_max_x_cm_for_y_cm(y_cm)

    def _cell_has_workplate_overlap(self, cell: GridCell) -> bool:
        """
        Conservative overlap check: if any sampled point in the cell is valid, keep the cell.
        """
        x_step = (cell.x_max_cm - cell.x_min_cm) / 20.0
        y_step = (cell.y_max_cm - cell.y_min_cm) / 20.0
        for iy in range(21):
            y_cm = cell.y_min_cm + iy * y_step
            for ix in range(21):
                x_cm = cell.x_min_cm + ix * x_step
                if self._is_within_env_workplate(x_cm, y_cm):
                    return True
        return False

    def _valid_grid_cells(self, config: DataGenerationConfig) -> list[GridCell]:
        """Return only cells that overlap the strict EnvControl workplate shape."""
        return [cell for cell in self.build_workplate_grid(config) if self._cell_has_workplate_overlap(cell)]

    def _rotation_intervals(self, config: DataGenerationConfig) -> list[tuple[float, float]]:
        """Split z-rotation range into configured intervals."""
        if not config.random_yaw:
            return [(0.0, 0.0)]

        min_rot = config.z_rotation_min_rad
        max_rot = config.z_rotation_max_rad

        if min_rot == max_rot:
            return [(min_rot, max_rot)]

        interval_size = (max_rot - min_rot) / config.z_rotation_intervals
        intervals: list[tuple[float, float]] = []
        start = min_rot
        for index in range(config.z_rotation_intervals):
            end = max_rot if index == config.z_rotation_intervals - 1 else start + interval_size
            intervals.append((start, end))
            start = end
        return intervals

    def count_data_points(self, config: DataGenerationConfig) -> int:
        """
        Return how many data points will be generated with the provided settings.
        """
        config.validate()
        valid_cells = len(self._valid_grid_cells(config))
        rotation_intervals = len(self._rotation_intervals(config))
        rotation_samples_per_interval = config.rotation_iteration_amount if config.random_yaw else 1
        return valid_cells * config.iteration_amount * rotation_intervals * rotation_samples_per_interval

    def _sample_valid_point_in_cell(
        self,
        cell: GridCell,
        rng: random.Random,
        max_attempts: int = 2000,
    ) -> tuple[float, float]:
        """Sample one valid point in a cell while respecting EnvControl bounds."""
        for _ in range(max_attempts):
            x_cm = rng.uniform(cell.x_min_cm, cell.x_max_cm)
            y_cm = rng.uniform(cell.y_min_cm, cell.y_max_cm)
            if self._is_within_env_workplate(x_cm, y_cm):
                return x_cm, y_cm
        raise RuntimeError(
            f"Could not find valid workplate point in cell {cell.cell_index} "
            f"after {max_attempts} attempts."
        )

    def _get_current_joint_rotations_rad(self) -> list[float]:
        """Read current robot joint rotations from the environment."""
        state = self.env.get_state(
            actuator_rotations=True,
            actuator_velocities=False,
            target_cube_state=False,
            graper=False,
            collisions=False,
            workplate_coverage=False,
            distance_to_target=False,
            image=False,
        )
        return [float(value) for value in state["actuator_rotations"]]

    @staticmethod
    def _wrapped_delta_deg(target_rad: float, current_rad: float) -> float:
        """Shortest signed angular delta in degrees in range [-180, 180)."""
        target_deg = target_rad * 180.0 / pi
        current_deg = current_rad * 180.0 / pi
        return ((target_deg - current_deg + 180.0) % 360.0) - 180.0

    def _move_to_joint_target(
        self,
        target_rotations_rad: list[float],
        tolerance_deg: float = 0.1,
        max_control_steps: int = 2000,
        grapper_state: bool = False,
        search_speed_multiplier: float = 1.0,
    ) -> None:
        """
        Move to target pose with synchronized arrival:
        the actuator with the longest normalized remaining move runs at full speed.
        """
        if len(target_rotations_rad) != 6:
            raise ValueError("target_rotations_rad must have 6 values")
        if search_speed_multiplier <= 0.0:
            raise ValueError("search_speed_multiplier must be > 0")

        effective_max_speeds = [speed * search_speed_multiplier for speed in self.ACTUATOR_MAX_SPEEDS_DEG_S]

        reached_target = False
        for _ in range(max_control_steps):
            current_rotations_rad = self._get_current_joint_rotations_rad()
            delta_deg = [
                self._wrapped_delta_deg(target, current)
                for target, current in zip(target_rotations_rad, current_rotations_rad)
            ]

            if max(abs(value) for value in delta_deg) <= tolerance_deg:
                reached_target = True
                break

            times_to_finish = [
                abs(delta) / max_speed if abs(delta) > tolerance_deg else 0.0
                for delta, max_speed in zip(delta_deg, effective_max_speeds)
            ]
            longest_time = max(times_to_finish)
            if longest_time <= 0.0:
                reached_target = True
                break

            velocities_deg_s = []
            for delta, max_speed, direction in zip(
                delta_deg,
                effective_max_speeds,
                self.ACTUATOR_STEP_DIRECTIONS,
            ):
                if abs(delta) <= tolerance_deg:
                    velocities_deg_s.append(0.0)
                    continue

                # delta = direction * velocity * time  =>  velocity = delta / (direction * time)
                velocity = delta / (direction * longest_time)

                # Stability guard: do not allow one control step to overshoot the target.
                max_without_overshoot = abs(delta) * self.CONTROL_FPS
                velocity_limit = min(max_speed, max_without_overshoot)

                if velocity > velocity_limit:
                    velocity = velocity_limit
                elif velocity < -velocity_limit:
                    velocity = -velocity_limit
                velocities_deg_s.append(velocity)

            if max(abs(value) for value in velocities_deg_s) <= 1e-6:
                reached_target = True
                break

            self.env.step(actuator_velocities=velocities_deg_s, grapper_state=grapper_state)

        if not reached_target:
            print("Warning: target pose not fully reached within max_control_steps")
        self._ensure_grapper_open()

    @staticmethod
    def build_dataset_filename(
        waypoint_index: int,
        joint_rotations_rad: list[float],
        cube_location_m: list[float],
        cube_z_rotation_rad: float,
        sample_index: int | None = None,
    ) -> str:
        """Build a user-readable and machine-parseable dataset filename."""
        if len(joint_rotations_rad) != 6:
            raise ValueError("joint_rotations_rad must contain 6 values")
        if len(cube_location_m) != 3:
            raise ValueError("cube_location_m must contain 3 values")

        parts = []
        if sample_index is not None:
            parts.append(f"s_{sample_index:06d}")
        parts.append(f"wp_{waypoint_index:04d}")
        parts.extend(f"j{index}_{value:.6f}" for index, value in enumerate(joint_rotations_rad))
        parts.extend(
            [
                f"cx_{cube_location_m[0]:.6f}",
                f"cy_{cube_location_m[1]:.6f}",
                f"cz_{cube_location_m[2]:.6f}",
                f"cyaw_{cube_z_rotation_rad:.6f}",
            ]
        )
        return "__".join(parts) + ".png"

    @staticmethod
    def parse_dataset_filename(file_name: str | Path) -> dict[str, Any]:
        """
        Parse labels encoded by `build_dataset_filename`.
        Useful for training-time target extraction.
        """
        stem = Path(file_name).stem
        values: dict[str, Any] = {}
        joints: dict[int, float] = {}

        for token in stem.split("__"):
            if "_" not in token:
                continue
            key, raw_value = token.split("_", 1)
            if key == "wp":
                values["waypoint_index"] = int(raw_value)
                continue
            if key == "s":
                values["sample_index"] = int(raw_value)
                continue
            if key.startswith("j") and key[1:].isdigit():
                joints[int(key[1:])] = float(raw_value)
                continue
            if key == "cx":
                values["cube_x_m"] = float(raw_value)
                continue
            if key == "cy":
                values["cube_y_m"] = float(raw_value)
                continue
            if key == "cz":
                values["cube_z_m"] = float(raw_value)
                continue
            if key == "cyaw":
                values["cube_z_rotation_rad"] = float(raw_value)

        if joints:
            values["joint_rotations_rad"] = [joints[index] for index in sorted(joints.keys())]

        return values

    def _ensure_grapper_open(self) -> None:
        """Force gripper open and keep it open."""
        self.env.step(actuator_velocities=[0.0] * 6, grapper_state=False)

    @staticmethod
    def _save_grayscale_png(image: list[list[float]], output_file: Path) -> None:
        """
        Save a grayscale image in PNG format using Pillow.
        """
        if not image or not image[0]:
            raise ValueError("image must be a non-empty 2D list")

        height = len(image)
        width = len(image[0])
        output_file.parent.mkdir(parents=True, exist_ok=True)

        pixels_uint8: list[int] = []
        for row in image:
            if len(row) != width:
                raise ValueError("image rows must all have equal length")
            for pixel in row:
                normalized = 0.0 if pixel < 0.0 else (1.0 if pixel > 1.0 else float(pixel))
                pixels_uint8.append(int(round(normalized * 255.0)))

        img = Image.new("L", (width, height))
        img.putdata(pixels_uint8)
        img.save(output_file, format="PNG")

    def move_along_search_path_to_cube(
        self,
        padding: float = 0.05,
        dataset_dir: str | Path | None = None,
        tolerance_deg: float = 0.1,
        max_control_steps_per_waypoint: int = 2000,
        search_speed_multiplier: float = 1.0,
        sample_index: int | None = None,
        quiet: bool = False,
    ) -> Path | None:
        """
        Follow the loaded search path and capture one labeled image when the cube is
        fully inside the camera view with the requested padding.
        """
        if not self.search_path:
            raise ValueError("Search path is empty. Provide waypoints in docs/search_path.json.")

        output_dir = Path(dataset_dir) if dataset_dir is not None else DEFAULT_CUBE_LOCALISATION_DATASET_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_grapper_open()

        for waypoint_index, waypoint_deg in enumerate(self.search_path):
            if len(waypoint_deg) != 6:
                raise ValueError(f"Expected 6 joint values per waypoint, got {len(waypoint_deg)} at index {waypoint_index}")

            waypoint_with_offset_deg = [
                float(value) + random.uniform(-2.0, 2.0)
                for value in waypoint_deg
            ]
            target_rotations_rad = [radians(value) for value in waypoint_with_offset_deg]
            self._move_to_joint_target(
                target_rotations_rad=target_rotations_rad,
                tolerance_deg=tolerance_deg,
                max_control_steps=max_control_steps_per_waypoint,
                grapper_state=False,
                search_speed_multiplier=search_speed_multiplier,
            )

            if not self.env.target_cube_within_padding(padding=padding):
                continue

            state = self.env.get_state(
                actuator_rotations=True,
                actuator_velocities=False,
                target_cube_state=True,
                graper=False,
                collisions=False,
                workplate_coverage=False,
                distance_to_target=False,
                image=True,
            )

            file_name = self.build_dataset_filename(
                waypoint_index=waypoint_index,
                joint_rotations_rad=[float(value) for value in state["actuator_rotations"]],
                cube_location_m=[float(value) for value in state["target_cube_location"]],
                cube_z_rotation_rad=float(state["target_cube_rotation"][2]),
                sample_index=sample_index,
            )
            output_path = output_dir / file_name
            self._save_grayscale_png(state["image"], output_path)
            if not quiet:
                print(f"Saved labeled image: {output_path}")
            return output_path

        if not quiet:
            print("Cube was not found within camera padding on the current search path.")
        return None

    def generate_cube_localisation_dataset(
        self,
        config: DataGenerationConfig,
        dataset_dir: str | Path | None = None,
        padding_min: float = 0.01,
        padding_max: float = 0.1,
        tolerance_deg: float = 0.1,
        max_control_steps_per_waypoint: int = 2000,
        search_speed_multiplier: float = 1.0,
        max_samples: int | None = None,
        stop_on_error: bool = False,
    ) -> dict[str, int]:
        """
        Full dataset generation pipeline:
        1) sample cube poses from grid/rotation config
        2) move robot along search path
        3) save one labeled image once cube satisfies camera padding rule
        """
        config.validate()
        if not (0.0 <= padding_min < 0.5 and 0.0 <= padding_max < 0.5):
            raise ValueError("padding_min and padding_max must be in [0.0, 0.5)")
        if padding_min > padding_max:
            raise ValueError("padding_min must be <= padding_max")

        output_dir = Path(dataset_dir) if dataset_dir is not None else DEFAULT_CUBE_LOCALISATION_DATASET_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        rng_padding = random.Random(config.seed)

        total_planned = self.count_data_points(config)
        limit = total_planned if max_samples is None else min(max_samples, total_planned)
        print(f"Starting dataset generation: planned={total_planned}, run_limit={limit}, output={output_dir}")

        saved = 0
        not_found = 0
        failed = 0
        processed = 0

        for planned_sample in self.iter_sampling_plan(config):
            if processed >= limit:
                break

            sample_index = int(planned_sample["sample_index"])
            cube_pose = planned_sample["cube_pose"]
            processed += 1

            try:
                self.env.reset(cube_position="home", robot_pose=config.robot_pose)
                self._ensure_grapper_open()
                self.env.set_cube_pose(
                    x=self._cm_to_m(cube_pose["x_cm"]),
                    y=self._cm_to_m(cube_pose["y_cm"]),
                    z=self._cm_to_m(cube_pose["z_cm"]),
                    yaw=cube_pose["yaw_rad"],
                )

                sample_padding = rng_padding.uniform(padding_min, padding_max)
                saved_path = self.move_along_search_path_to_cube(
                    padding=sample_padding,
                    dataset_dir=output_dir,
                    tolerance_deg=tolerance_deg,
                    max_control_steps_per_waypoint=max_control_steps_per_waypoint,
                    search_speed_multiplier=search_speed_multiplier,
                    sample_index=sample_index,
                    quiet=True,
                )
                if saved_path is None:
                    not_found += 1
                    print(
                        f"[{processed}/{limit}] sample={sample_index} "
                        f"padding={sample_padding:.4f} not found in padded camera view"
                    )
                else:
                    saved += 1
                    print(
                        f"[{processed}/{limit}] sample={sample_index} "
                        f"padding={sample_padding:.4f} saved={saved_path.name}"
                    )
            except Exception as exc:
                failed += 1
                print(f"[{processed}/{limit}] sample={sample_index} failed: {exc}")
                if stop_on_error:
                    raise

        summary = {
            "planned": total_planned,
            "processed": processed,
            "saved": saved,
            "not_found": not_found,
            "failed": failed,
        }
        print("Generation summary:", summary)
        return summary

    def iter_sampling_plan(self, config: DataGenerationConfig) -> Iterable[dict[str, Any]]:
        """
        Yield random placements where each grid cell is sampled `iteration_amount` times.
        """
        rng = random.Random(config.seed)
        grid_cells = self._valid_grid_cells(config)
        rotation_intervals = self._rotation_intervals(config)
        sample_index = 0

        for cell in grid_cells:
            for iteration in range(config.iteration_amount):
                x_cm, y_cm = self._sample_valid_point_in_cell(cell=cell, rng=rng)
                for interval_index, (interval_min, interval_max) in enumerate(rotation_intervals):
                    rotation_samples = config.rotation_iteration_amount if config.random_yaw else 1
                    for rotation_iteration in range(rotation_samples):
                        if config.random_yaw:
                            yaw_rad = interval_min if interval_min == interval_max else rng.uniform(interval_min, interval_max)
                        else:
                            yaw_rad = 0.0

                        yield {
                            "sample_index": sample_index,
                            "iteration_in_cell": iteration,
                            "rotation_interval_index": interval_index,
                            "rotation_iteration_in_interval": rotation_iteration,
                            "rotation_interval_min_rad": interval_min,
                            "rotation_interval_max_rad": interval_max,
                            "grid_cell": asdict(cell),
                            "cube_pose": {
                                "x_cm": x_cm,
                                "y_cm": y_cm,
                                "z_cm": config.cube_z_cm,
                                "yaw_rad": yaw_rad,
                            },
                        }
                        sample_index += 1

    def generate(self, config: DataGenerationConfig) -> list[dict[str, Any]]:
        """
        Run the configured grid sampling and return generated sample records.

        This is intentionally a scaffold: adapt capture payload and storage format
        according to your training pipeline requirements.
        """
        records: list[dict[str, Any]] = []
        for planned_sample in self.iter_sampling_plan(config):
            if config.reset_before_each_sample:
                self.env.reset(cube_position="home", robot_pose=config.robot_pose)

            cube_pose = planned_sample["cube_pose"]
            self.env.set_cube_pose(
                x=self._cm_to_m(cube_pose["x_cm"]),
                y=self._cm_to_m(cube_pose["y_cm"]),
                z=self._cm_to_m(cube_pose["z_cm"]),
                yaw=cube_pose["yaw_rad"],
            )

            state = self.env.get_state(
                actuator_rotations=True,
                actuator_velocities=True,
                target_cube_state=True,
                graper=True,
                collisions=True,
                workplate_coverage=True,
                distance_to_target=True,
                image=config.include_image,
            )

            record = dict(planned_sample)
            record["state"] = state
            records.append(record)

        return records

    def run_grid_test(
        self,
        config: DataGenerationConfig,
        delay_seconds: float = 0.1,
        positions_per_cell: int = 2,
    ) -> None:
        """
        Test-only traversal: move the cube through each grid cell without generating data.
        """
        if positions_per_cell != 2:
            raise ValueError("This test scaffold currently supports exactly 2 positions per cell.")

        self.env.reset(cube_position="home", robot_pose=config.robot_pose)
        cells = self._valid_grid_cells(config)
        total_positions = len(cells) * positions_per_cell
        moved_positions = 0

        for cell in cells:
            cell_rng = random.Random(cell.cell_index)
            test_positions = [
                self._sample_valid_point_in_cell(cell=cell, rng=cell_rng),
                self._sample_valid_point_in_cell(cell=cell, rng=cell_rng),
            ]
            if test_positions[0] == test_positions[1]:
                test_positions[1] = self._sample_valid_point_in_cell(cell=cell, rng=cell_rng)

            for pos_idx, (x_cm, y_cm) in enumerate(test_positions, start=1):
                self.env.set_cube_pose(
                    x=self._cm_to_m(x_cm),
                    y=self._cm_to_m(y_cm),
                    z=self._cm_to_m(config.cube_z_cm),
                    yaw=0.0,
                )
                moved_positions += 1
                print(
                    f"[{moved_positions}/{total_positions}] "
                    f"cell={cell.cell_index} pos={pos_idx}/2 x_cm={x_cm:.2f} y_cm={y_cm:.2f}"
                )
                time.sleep(delay_seconds)

    def run_sampling_plan_test(self, config: DataGenerationConfig, delay_seconds: float = 0.1) -> None:
        """
        Test-only traversal of the full sampling plan (position + z-rotation settings).
        No data is collected.
        """
        total_samples = self.count_data_points(config)
        self.env.reset(cube_position="home", robot_pose=config.robot_pose)

        for idx, planned_sample in enumerate(self.iter_sampling_plan(config), start=1):
            if config.reset_before_each_sample:
                self.env.reset(cube_position="home", robot_pose=config.robot_pose)

            cube_pose = planned_sample["cube_pose"]
            self.env.set_cube_pose(
                x=self._cm_to_m(cube_pose["x_cm"]),
                y=self._cm_to_m(cube_pose["y_cm"]),
                z=self._cm_to_m(cube_pose["z_cm"]),
                yaw=cube_pose["yaw_rad"],
            )

            print(
                f"[{idx}/{total_samples}] "
                f"cell={planned_sample['grid_cell']['cell_index']} "
                f"pos_it={planned_sample['iteration_in_cell'] + 1}/{config.iteration_amount} "
                f"rot_interval={planned_sample['rotation_interval_index'] + 1}/{len(self._rotation_intervals(config))} "
                f"rot_it={planned_sample['rotation_iteration_in_interval'] + 1}/"
                f"{config.rotation_iteration_amount if config.random_yaw else 1} "
                f"yaw_rad={cube_pose['yaw_rad']:.4f}"
            )
            time.sleep(delay_seconds)

    @staticmethod
    def export_jsonl(records: Iterable[dict[str, Any]], output_file: str | Path) -> Path:
        """Write records as JSONL for downstream training pipelines."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record) + "\n")

        return output_path


if __name__ == "__main__":
    # -----------------------
    # Editable runtime config
    # -----------------------
    RUN_MODE = "generate_dataset"  # options: "generate_dataset", "sampling_plan_test", "search_path_test"
    HOST = "localhost"
    PORT = 5055
    TIMEOUT = 5.0

    DATASET_DIR = DEFAULT_CUBE_LOCALISATION_DATASET_DIR
    CAMERA_PADDING = 0.05
    CAMERA_PADDING_MIN = 0.05
    CAMERA_PADDING_MAX = 0.2
    JOINT_TOLERANCE_DEG = 0.1
    MAX_CONTROL_STEPS_PER_WAYPOINT = 2000
    SEARCH_SPEED_MULTIPLIER = 5.0  # 1.0 = baseline, 2.0 = ~2x, 3.0 = ~3x
    TEST_DELAY_SECONDS = 0.1
    MAX_SAMPLES = None  # e.g. 100 for a short run
    STOP_ON_ERROR = False

    config = DataGenerationConfig(
        box_size_cm=5.0,
        iteration_amount=2,
        cube_z_cm=2.5,
        random_yaw=True,
        z_rotation_min_rad=-pi / 4,
        z_rotation_max_rad=pi / 4,
        z_rotation_intervals=4,
        rotation_iteration_amount=2,
        robot_pose="home",
        reset_before_each_sample=False,
        include_image=False,
        seed=None,
    )

    with EnvInteface(host=HOST, port=PORT, timeout=TIMEOUT) as env:
        generator = DataGenerator(env)
        print("counted data points to generate:", generator.count_data_points(config))

        if RUN_MODE == "search_path_test":
            saved_file = generator.move_along_search_path_to_cube(
                padding=CAMERA_PADDING,
                dataset_dir=DATASET_DIR,
                tolerance_deg=JOINT_TOLERANCE_DEG,
                max_control_steps_per_waypoint=MAX_CONTROL_STEPS_PER_WAYPOINT,
                search_speed_multiplier=SEARCH_SPEED_MULTIPLIER,
            )
            if saved_file is not None:
                print("Parsed labels:", generator.parse_dataset_filename(saved_file.name))
        elif RUN_MODE == "sampling_plan_test":
            generator.run_sampling_plan_test(
                config=config,
                delay_seconds=TEST_DELAY_SECONDS,
            )
        elif RUN_MODE == "generate_dataset":
            generator.generate_cube_localisation_dataset(
                config=config,
                dataset_dir=DATASET_DIR,
                padding_min=CAMERA_PADDING_MIN,
                padding_max=CAMERA_PADDING_MAX,
                tolerance_deg=JOINT_TOLERANCE_DEG,
                max_control_steps_per_waypoint=MAX_CONTROL_STEPS_PER_WAYPOINT,
                search_speed_multiplier=SEARCH_SPEED_MULTIPLIER,
                max_samples=MAX_SAMPLES,
                stop_on_error=STOP_ON_ERROR,
            )
        else:
            raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")
