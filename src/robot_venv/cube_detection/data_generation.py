"""Generate cube-detection training images inside Cube_detection_env.blend."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import bpy


def _resolve_script_dir() -> Path:
    """
    Resolve the directory that contains `cube_detection_generator.py`.

    This is robust for Blender execution contexts where module search paths can differ.
    """
    candidates: list[Path] = []

    file_path = globals().get("__file__")
    if file_path:
        candidates.append(Path(file_path).resolve().parent)

    text = getattr(getattr(bpy.context, "space_data", None), "text", None)
    if text is not None and getattr(text, "filepath", ""):
        candidates.append(Path(bpy.path.abspath(text.filepath)).resolve().parent)

    if bpy.data.filepath:
        blend_dir = Path(bpy.path.abspath("//")).resolve()
        candidates.append(blend_dir / "src" / "robot_venv" / "cube_detection")

    candidates.append(Path.cwd())

    for candidate in candidates:
        if (candidate / "cube_detection_generator.py").exists():
            return candidate

    raise ModuleNotFoundError(
        "Could not locate cube_detection_generator.py. "
        "Ensure data_generation.py and cube_detection_generator.py are in the same folder, "
        "or run from the project where src/robot_venv/cube_detection exists."
    )


def _load_cube_detection_generator(script_dir: Path):
    module_path = script_dir / "cube_detection_generator.py"
    spec = importlib.util.spec_from_file_location("cube_detection_generator", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CubeDetectionGenerator


SCRIPT_DIR = _resolve_script_dir()
CubeDetectionGenerator = _load_cube_detection_generator(SCRIPT_DIR)


PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "Cube_Detection_dataset"


def _clear_existing_pngs(output_dir: Path) -> int:
    deleted = 0
    for png_file in output_dir.glob("*.png"):
        png_file.unlink(missing_ok=True)
        deleted += 1
    return deleted


def _save_metadata_jsonl(records: list[dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def _summarize(records: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "frames_total": len(records),
        "cube_visible_frames": 0,
        "no_cube_visible_frames": 0,
        "samples_total": 0,
        "samples_with_terminal_no_cube": 0,
    }
    if not records:
        return summary

    per_sample: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        sample_idx = int(record["sample_index"])
        per_sample.setdefault(sample_idx, []).append(record)
        if record["label"] == "cube_visible":
            summary["cube_visible_frames"] += 1
        elif record["label"] == "no_cube_visible":
            summary["no_cube_visible_frames"] += 1

    summary["samples_total"] = len(per_sample)

    terminal_count = 0
    for sample_idx, sample_records in per_sample.items():
        last_record = max(sample_records, key=lambda item: int(item["frame_index"]))
        if last_record["label"] == "no_cube_visible":
            terminal_count += 1
        else:
            print(
                f"Warning: sample {sample_idx} has no terminal no_cube_visible frame in saved records."
            )
    summary["samples_with_terminal_no_cube"] = terminal_count
    return summary


if __name__ == "__main__":
    # -----------------------
    # Editable runtime config
    # -----------------------
    OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    N_SAMPLES = 300
    MAX_STEPS_PER_SAMPLE = 30
    SEED = None

    # If True, delete existing PNG files in OUTPUT_DIR before generation.
    CLEAR_EXISTING_PNGS = False

    # If True, write one JSONL row per generated frame to OUTPUT_DIR/labels.jsonl
    SAVE_METADATA_JSONL = True

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if CLEAR_EXISTING_PNGS:
        deleted = _clear_existing_pngs(output_dir=output_dir)
        print(f"Deleted {deleted} existing PNG files from {output_dir}")

    generator = CubeDetectionGenerator()
    records = generator.generate_samples(
        n_samples=N_SAMPLES,
        output_dir=output_dir,
        max_steps_per_sample=MAX_STEPS_PER_SAMPLE,
        seed=SEED,
    )

    if SAVE_METADATA_JSONL:
        metadata_file = output_dir / "labels.jsonl"
        _save_metadata_jsonl(records=records, output_file=metadata_file)
        print(f"Saved metadata: {metadata_file}")

    summary = _summarize(records=records)
    print("Generation complete.")
    print(json.dumps(summary, indent=2))
