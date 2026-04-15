from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "hm3d_usd"
DEFAULT_COLLISION_MODE = "triangle_mesh"
SUPPORTED_COLLISION_MODES = ("no_collision", "convex_hull", "triangle_mesh")
LEGACY_COLLISION_MODE_ALIASES = {"none": "no_collision"}


def validate_input_mode(
    input_glb: str | Path | None, input_dir: str | Path | None
) -> tuple[Path | None, Path | None]:
    """Normalize and validate the mutually exclusive input modes."""

    has_glb = input_glb is not None
    has_dir = input_dir is not None
    if has_glb == has_dir:
        raise ValueError("Exactly one of input_glb or input_dir must be provided.")

    if has_glb:
        glb_path = Path(input_glb).expanduser().resolve()
        if glb_path.suffix.lower() != ".glb":
            raise ValueError(f"input_glb must point to a .glb file: {glb_path}")
        if not glb_path.is_file():
            raise ValueError(f"input_glb does not exist or is not a file: {glb_path}")
        return glb_path, None

    scene_root = Path(input_dir).expanduser().resolve()
    if not scene_root.is_dir():
        raise ValueError(f"input_dir does not exist or is not a directory: {scene_root}")
    return None, scene_root


def resolve_collision_mode(collision_mode: str) -> str:
    """Validate and normalize collision mode."""

    normalized = collision_mode.strip().lower()
    if normalized == "mesh_simplification":
        raise ValueError(
            "Unsupported collision_mode 'mesh_simplification'. "
            "The current Isaac Lab mesh collision mapping writes an invalid USD "
            "physics:approximation token for this mode. Use 'triangle_mesh' instead."
        )
    normalized = LEGACY_COLLISION_MODE_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_COLLISION_MODES:
        raise ValueError(
            f"Unsupported collision_mode '{collision_mode}'. "
            f"Expected one of: {', '.join(SUPPORTED_COLLISION_MODES)}."
        )
    return normalized


def _iter_scene_dirs(scene_root: Path) -> list[Path]:
    return sorted(path for path in scene_root.iterdir() if path.is_dir())


def find_invalid_scene_dirs(scene_root: str | Path) -> list[str]:
    """Report HM3D scene directories that do not contain exactly one .glb file."""

    root = Path(scene_root)
    errors: list[str] = []
    for scene_dir in _iter_scene_dirs(root):
        glb_paths = sorted(scene_dir.glob("*.glb"))
        if len(glb_paths) == 1:
            continue
        if len(glb_paths) == 0:
            errors.append(f"{scene_dir.name}: expected exactly one .glb file, found none.")
        else:
            errors.append(f"{scene_dir.name}: expected exactly one .glb file, found {len(glb_paths)}.")
    return errors


def discover_glb_files(input_glb: str | Path | None, input_dir: str | Path | None) -> list[Path]:
    """Discover valid HM3D GLB scene files from the selected input mode."""

    glb_path, scene_root = validate_input_mode(input_glb=input_glb, input_dir=input_dir)
    if glb_path is not None:
        return [glb_path]

    assert scene_root is not None
    scenes: list[Path] = []
    for scene_dir in _iter_scene_dirs(scene_root):
        glb_paths = sorted(scene_dir.glob("*.glb"))
        if len(glb_paths) == 1:
            scenes.append(glb_paths[0])
    return scenes


def derive_output_paths(glb_path: str | Path, output_root: str | Path) -> tuple[Path, str]:
    """Derive the output USD directory and file name from a GLB path."""

    scene_path = Path(glb_path)
    scene_name = scene_path.stem
    usd_dir = Path(output_root).expanduser().resolve() / scene_name
    return usd_dir, f"{scene_name}.usd"


def _create_mesh_collision_cfg(collision_mode: str) -> tuple[Any | None, Any | None]:
    from isaaclab.sim.schemas import schemas_cfg

    normalized = resolve_collision_mode(collision_mode)
    if normalized == "no_collision":
        return None, None
    if normalized == "convex_hull":
        return schemas_cfg.CollisionPropertiesCfg(collision_enabled=True), schemas_cfg.ConvexHullPropertiesCfg()
    return (
        schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
        schemas_cfg.TriangleMeshPropertiesCfg(),
    )


def build_converter_cfg(
    glb_path: str | Path,
    output_root: str | Path,
    force: bool,
    make_instanceable: bool,
    collision_mode: str,
):
    """Build an Isaac Lab MeshConverterCfg for one HM3D scene."""

    from isaaclab.sim.converters import MeshConverterCfg

    usd_dir, usd_file_name = derive_output_paths(glb_path, output_root)
    collision_props, mesh_collision_props = _create_mesh_collision_cfg(collision_mode)
    return MeshConverterCfg(
        asset_path=str(Path(glb_path).expanduser().resolve()),
        usd_dir=str(usd_dir),
        usd_file_name=usd_file_name,
        force_usd_conversion=force,
        make_instanceable=make_instanceable,
        collision_props=collision_props,
        mesh_collision_props=mesh_collision_props,
    )


def convert_one_scene(
    glb_path: str | Path,
    output_root: str | Path,
    force: bool,
    make_instanceable: bool,
    collision_mode: str,
    converter_factory=None,
) -> Path:
    """Convert one GLB scene and return the generated USD path."""

    if converter_factory is None:
        from isaaclab.sim.converters import MeshConverter

        converter_factory = MeshConverter

    cfg = build_converter_cfg(
        glb_path=glb_path,
        output_root=output_root,
        force=force,
        make_instanceable=make_instanceable,
        collision_mode=collision_mode,
    )
    Path(cfg.usd_dir).mkdir(parents=True, exist_ok=True)
    converter = converter_factory(cfg)
    return Path(converter.usd_path)


def convert_scenes(
    scene_paths: list[Path],
    output_root: str | Path,
    force: bool,
    make_instanceable: bool,
    collision_mode: str,
    converter_factory=None,
) -> dict[str, list[str]]:
    """Convert multiple scenes and collect success or failure summaries."""

    summary = {"converted": [], "failed": []}
    for glb_path in scene_paths:
        try:
            usd_path = convert_one_scene(
                glb_path=glb_path,
                output_root=output_root,
                force=force,
                make_instanceable=make_instanceable,
                collision_mode=collision_mode,
                converter_factory=converter_factory,
            )
        except Exception as error:  # pragma: no cover - runtime failures depend on Isaac Sim.
            summary["failed"].append(f"{Path(glb_path).stem}: {error}")
            continue

        summary["converted"].append(str(usd_path))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args without importing Isaac Sim dependent modules at module import time."""

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Convert HM3D GLB scenes into USD assets for Isaac Lab.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_glb", type=Path, default=None, help="Path to a single HM3D .glb scene file.")
    input_group.add_argument("--input_dir", type=Path, default=None, help="Path to an HM3D split directory.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory for generated USD assets.",
    )
    parser.add_argument("--force", action="store_true", default=False, help="Force reconversion of USD assets.")
    parser.add_argument(
        "--make_instanceable",
        action="store_true",
        default=False,
        help="Generate instanceable USD geometry.",
    )
    parser.add_argument(
        "--collision_mode",
        type=str,
        default=DEFAULT_COLLISION_MODE,
        help=f"Collision mode: {', '.join(SUPPORTED_COLLISION_MODES)}.",
    )
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args(argv)
    args.input_glb, args.input_dir = validate_input_mode(args.input_glb, args.input_dir)
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    args.collision_mode = resolve_collision_mode(args.collision_mode)
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point for runtime conversion."""

    args = parse_args(argv)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        invalid_dirs = find_invalid_scene_dirs(args.input_dir) if args.input_dir is not None else []
        scene_paths = discover_glb_files(args.input_glb, args.input_dir)
        if not scene_paths:
            raise ValueError("No valid HM3D .glb scenes were found for conversion.")

        summary = convert_scenes(
            scene_paths=scene_paths,
            output_root=args.output_dir,
            force=args.force,
            make_instanceable=args.make_instanceable,
            collision_mode=args.collision_mode,
        )
        summary["failed"].extend(invalid_dirs)

        for usd_path in summary["converted"]:
            print(f"[INFO] Converted scene to: {usd_path}")
        for error in summary["failed"]:
            print(f"[ERROR] {error}")

        print(
            "[INFO] Conversion summary: "
            f"{len(summary['converted'])} succeeded, {len(summary['failed'])} failed."
        )

        if summary["failed"]:
            raise RuntimeError(f"Failed to convert {len(summary['failed'])} scene(s).")
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
