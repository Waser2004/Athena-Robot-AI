"""Search-path based cube detection data generation."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEARCH_PATH_FILE = PROJECT_ROOT / "docs" / "search_path.json"
DEFAULT_CUBE_DETECTION_DATASET_DIR = PROJECT_ROOT / "docs" / "Cube_Detection_dataset"
PREFERRED_BLEND_FILE = PROJECT_ROOT / "src" / "robot_venv" / "Robot_V2_venv.blend"
FALLBACK_BLEND_FILE = PROJECT_ROOT / "src" / "robot_venv" / "Robot_V2_env.blend"


def resolve_required_blend_file() -> Path:
    """Resolve the preferred Robot V2 blend file for this generator."""
    if PREFERRED_BLEND_FILE.exists():
        return PREFERRED_BLEND_FILE
    if FALLBACK_BLEND_FILE.exists():
        return FALLBACK_BLEND_FILE
    return PREFERRED_BLEND_FILE


def load_search_path(path: Path | None = None) -> list[list[float]]:
    """Load search-path waypoints from JSON."""
    search_path_file = path or DEFAULT_SEARCH_PATH_FILE
    with search_path_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        payload = payload.get("search_path", [])

    return [list(map(float, waypoint)) for waypoint in payload]


@dataclass(frozen=True)
class DataGenerationConfig:
    iteration_amount: int
    robot_pose: str = "home"
    cube_z_m: float = 0.025
    random_yaw: bool = True
    yaw_min_rad: float = -pi
    yaw_max_rad: float = pi
    tolerance_deg: float = 0.1
    max_control_steps_per_waypoint: int = 2000
    search_speed_multiplier: float = 1.0
    min_waypoint_step_jump: int = 1
    max_waypoint_step_jump: int = 10
    reset_after_each_cycle: bool = True
    seed: int | None = None

    def validate(self) -> None:
        if self.iteration_amount <= 0:
            raise ValueError("iteration_amount must be > 0")
        if self.cube_z_m <= 0.0:
            raise ValueError("cube_z_m must be > 0")
        if self.yaw_min_rad > self.yaw_max_rad:
            raise ValueError("yaw_min_rad must be <= yaw_max_rad")
        if self.tolerance_deg <= 0.0:
            raise ValueError("tolerance_deg must be > 0")
        if self.max_control_steps_per_waypoint <= 0:
            raise ValueError("max_control_steps_per_waypoint must be > 0")
        if self.search_speed_multiplier <= 0.0:
            raise ValueError("search_speed_multiplier must be > 0")
        if self.min_waypoint_step_jump <= 0:
            raise ValueError("min_waypoint_step_jump must be > 0")
        if self.max_waypoint_step_jump < self.min_waypoint_step_jump:
            raise ValueError("max_waypoint_step_jump must be >= min_waypoint_step_jump")


class DataGenerator:
    """Generate cube-detection images by traversing robot search-path checkpoints."""

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
    def _workplate_x_max_for_y(y_m: float) -> float:
        if y_m <= -0.16:
            return 0.05

        radial = (-(y_m * 1000.0) - 25.0)
        radicand = (135.0**2) - (radial**2)
        if radicand < 0.0:
            return -0.24
        return -(sqrt(radicand) + 25.0) / 1000.0

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

    def _sample_cube_pose(self, config: DataGenerationConfig, rng: random.Random) -> tuple[float, float, float, float]:
        x_m, y_m = self._sample_valid_workplate_xy(rng=rng)
        yaw_rad = 0.0
        if config.random_yaw:
            if config.yaw_min_rad == config.yaw_max_rad:
                yaw_rad = float(config.yaw_min_rad)
            else:
                yaw_rad = rng.uniform(config.yaw_min_rad, config.yaw_max_rad)
        return x_m, y_m, float(config.cube_z_m), float(yaw_rad)

    def _get_current_joint_rotations_rad(self) -> list[float]:
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
        target_deg = target_rad * 180.0 / pi
        current_deg = current_rad * 180.0 / pi
        return ((target_deg - current_deg + 180.0) % 360.0) - 180.0

    def _move_to_joint_target(
        self,
        target_rotations_rad: list[float],
        tolerance_deg: float,
        max_control_steps: int,
        search_speed_multiplier: float,
    ) -> None:
        if len(target_rotations_rad) != 6:
            raise ValueError("target_rotations_rad must have 6 values")

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

            self.env.step(actuator_velocities=velocities_deg_s, grapper_state=False)

        if not reached_target:
            print("Warning: target pose not fully reached within max_control_steps")
        self._ensure_grapper_open()

    def _ensure_grapper_open(self) -> None:
        self.env.step(actuator_velocities=[0.0] * 6, grapper_state=False)

    @staticmethod
    def _save_grayscale_png(image: list[list[float]], output_file: Path) -> None:
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

    @staticmethod
    def build_dataset_filename(
        sample_index: int,
        frame_index: int,
        label: str,
        visible_image_ratio: float,
        inframe_fraction: float,
        edge_margin: float,
    ) -> str:
        return (
            f"s_{sample_index:06d}__f_{frame_index:04d}__label_{label}"
            f"__vis_{visible_image_ratio:.6f}"
            f"__infrm_{inframe_fraction:.6f}"
            f"__edge_{edge_margin:.6f}.png"
        )

    @staticmethod
    def parse_dataset_filename(file_name: str | Path) -> dict[str, Any]:
        stem = Path(file_name).stem
        values: dict[str, Any] = {}

        for token in stem.split("__"):
            if "_" not in token:
                continue
            key, raw_value = token.split("_", 1)
            if key == "s":
                values["sample_index"] = int(raw_value)
            elif key == "f":
                values["frame_index"] = int(raw_value)
            elif key == "label":
                values["label"] = raw_value
            elif key == "vis":
                values["visible_image_ratio"] = float(raw_value)
            elif key == "infrm":
                values["inframe_fraction"] = float(raw_value)
            elif key == "edge":
                values["edge_margin"] = float(raw_value)

        return values

    def _get_visibility_labels(self) -> dict[str, Any]:
        if hasattr(self.env, "cube_visibility_labels"):
            labels = self.env.cube_visibility_labels()
        else:
            labels = self.env.call(function="cube_visibility_labels", args={}, expect_response=True)

        required = ("label", "cube_visible", "visible_image_ratio", "inframe_fraction", "edge_margin")
        missing = [key for key in required if key not in labels]
        if missing:
            raise RuntimeError(
                "Environment response for cube visibility labels is missing keys: "
                + ", ".join(missing)
            )
        return labels

    def count_data_points(self, config: DataGenerationConfig) -> int:
        config.validate()
        if not self.search_path:
            raise ValueError("Search path is empty. Provide waypoints in docs/search_path.json.")
        # Upper bound (dense stepping with jump size = 1).
        return len(self.search_path) * config.iteration_amount

    def iter_sampling_plan(self, config: DataGenerationConfig) -> Iterable[dict[str, int]]:
        config.validate()
        if not self.search_path:
            raise ValueError("Search path is empty. Provide waypoints in docs/search_path.json.")

        sample_index = 0
        for iteration_index in range(config.iteration_amount):
            for waypoint_index in range(len(self.search_path)):
                yield {
                    "sample_index": sample_index,
                    "iteration_index": iteration_index,
                    "waypoint_index": waypoint_index,
                }
                sample_index += 1

    def generate_cube_detection_dataset(
        self,
        config: DataGenerationConfig,
        dataset_dir: str | Path | None = None,
        max_samples: int | None = None,
        stop_on_error: bool = False,
    ) -> tuple[dict[str, int], list[dict[str, Any]]]:
        config.validate()
        if not self.search_path:
            raise ValueError("Search path is empty. Provide waypoints in docs/search_path.json.")

        output_dir = Path(dataset_dir) if dataset_dir is not None else DEFAULT_CUBE_DETECTION_DATASET_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        planned = self.count_data_points(config)
        run_limit = planned if max_samples is None else min(max_samples, planned)

        print(
            "Starting cube detection dataset generation: "
            f"planned={planned}, run_limit={run_limit}, output={output_dir}"
        )

        rng = random.Random(config.seed)
        records: list[dict[str, Any]] = []

        processed = 0
        saved = 0
        failed = 0

        # Initial reset before first cycle.
        self.env.reset(cube_position="home", robot_pose=config.robot_pose)
        self._ensure_grapper_open()

        for iteration_index in range(config.iteration_amount):
            if processed >= run_limit:
                break

            waypoint_index = 0
            while waypoint_index < len(self.search_path):
                if processed >= run_limit:
                    break

                processed += 1
                sample_index = processed - 1

                try:
                    waypoint_deg = self.search_path[waypoint_index]
                    if len(waypoint_deg) != 6:
                        raise ValueError(
                            "Expected 6 joint values per waypoint, "
                            f"got {len(waypoint_deg)} at index {waypoint_index}"
                        )

                    target_rotations_rad = [radians(float(value)) for value in waypoint_deg]
                    self._move_to_joint_target(
                        target_rotations_rad=target_rotations_rad,
                        tolerance_deg=config.tolerance_deg,
                        max_control_steps=config.max_control_steps_per_waypoint,
                        search_speed_multiplier=config.search_speed_multiplier,
                    )

                    # Per requirement: after each robot move to a checkpoint,
                    # randomly place the cube on the workplate.
                    cube_x_m, cube_y_m, cube_z_m, cube_yaw_rad = self._sample_cube_pose(config=config, rng=rng)
                    self.env.set_cube_pose(
                        x=cube_x_m,
                        y=cube_y_m,
                        z=cube_z_m,
                        yaw=cube_yaw_rad,
                    )

                    labels = self._get_visibility_labels()
                    state = self.env.get_state(
                        actuator_rotations=True,
                        actuator_velocities=False,
                        target_cube_state=False,
                        graper=False,
                        collisions=False,
                        workplate_coverage=False,
                        distance_to_target=False,
                        image=True,
                    )

                    file_name = self.build_dataset_filename(
                        sample_index=sample_index,
                        frame_index=waypoint_index,
                        label=str(labels["label"]),
                        visible_image_ratio=float(labels["visible_image_ratio"]),
                        inframe_fraction=float(labels["inframe_fraction"]),
                        edge_margin=float(labels["edge_margin"]),
                    )
                    output_path = output_dir / file_name
                    self._save_grayscale_png(state["image"], output_path)

                    record = {
                        "sample_index": sample_index,
                        "iteration_index": iteration_index,
                        "waypoint_index": waypoint_index,
                        "file_path": str(output_path),
                        "label": str(labels["label"]),
                        "cube_visible": bool(labels["cube_visible"]),
                        "visible_image_ratio": float(labels["visible_image_ratio"]),
                        "inframe_fraction": float(labels["inframe_fraction"]),
                        "edge_margin": float(labels["edge_margin"]),
                        "cube_pose": {
                            "x_m": float(cube_x_m),
                            "y_m": float(cube_y_m),
                            "z_m": float(cube_z_m),
                            "yaw_rad": float(cube_yaw_rad),
                        },
                    }
                    records.append(record)
                    saved += 1
                    print(
                        f"[{processed}/{run_limit}] "
                        f"iter={iteration_index + 1}/{config.iteration_amount} "
                        f"wp={waypoint_index + 1}/{len(self.search_path)} "
                        f"saved={output_path.name}"
                    )
                except Exception as exc:
                    failed += 1
                    print(
                        f"[{processed}/{run_limit}] "
                        f"iter={iteration_index + 1}/{config.iteration_amount} "
                        f"wp={waypoint_index + 1}/{len(self.search_path)} "
                        f"failed: {exc}"
                    )
                    if stop_on_error:
                        raise

                step_jump = rng.randint(config.min_waypoint_step_jump, config.max_waypoint_step_jump)
                waypoint_index += step_jump

            # Per requirement: reset environment after a completed search cycle
            # so lighting changes for the next cycle.
            if config.reset_after_each_cycle and iteration_index < config.iteration_amount - 1:
                self.env.reset(cube_position="home", robot_pose=config.robot_pose)
                self._ensure_grapper_open()

        summary = {
            "planned": planned,
            "processed": processed,
            "saved": saved,
            "failed": failed,
        }
        print("Generation summary:", summary)
        return summary, records

    @staticmethod
    def export_jsonl(records: Iterable[dict[str, Any]], output_file: str | Path) -> Path:
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
    HOST = "localhost"
    PORT = 5055
    TIMEOUT = 5.0

    DATASET_DIR = DEFAULT_CUBE_DETECTION_DATASET_DIR
    ITERATION_AMOUNT = 1000
    MAX_SAMPLES = None
    STOP_ON_ERROR = False
    SAVE_METADATA_JSONL = True

    config = DataGenerationConfig(
        iteration_amount=ITERATION_AMOUNT,
        robot_pose="home",
        cube_z_m=0.025,
        random_yaw=True,
        yaw_min_rad=-pi,
        yaw_max_rad=pi,
        tolerance_deg=0.1,
        max_control_steps_per_waypoint=2000,
        search_speed_multiplier=5.0,
        min_waypoint_step_jump=1,
        max_waypoint_step_jump=10,
        reset_after_each_cycle=True,
        seed=None,
    )

    blend_file = resolve_required_blend_file()
    print(
        "Expected Blender scene for this generator: "
        f"{blend_file}"
    )

    with EnvInteface(host=HOST, port=PORT, timeout=TIMEOUT) as env:
        generator = DataGenerator(env)
        print("counted data points to generate:", generator.count_data_points(config))

        summary, records = generator.generate_cube_detection_dataset(
            config=config,
            dataset_dir=DATASET_DIR,
            max_samples=MAX_SAMPLES,
            stop_on_error=STOP_ON_ERROR,
        )

        if SAVE_METADATA_JSONL:
            metadata_file = Path(DATASET_DIR) / "labels.jsonl"
            saved_jsonl = generator.export_jsonl(records=records, output_file=metadata_file)
            print(f"Saved metadata: {saved_jsonl}")

        print("Generation complete:", summary)
