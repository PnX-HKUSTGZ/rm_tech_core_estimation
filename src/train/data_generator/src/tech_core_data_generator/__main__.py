"""@brief Generate Blender training data, MuJoCo holdouts or a single labelled preview."""

import argparse
import sys
from pathlib import Path

from PIL import Image

from .camera import CameraConfig, Viewpoint
from .dataset import generate_dataset, require_empty
from .lighting import LightState, Team
from .simulator import TechCoreSimulator


def main(argv: list[str] | None = None):
    """@brief Render the dataset or preview requested on the command line.

    @param[in] argv Arguments after the module name, or None to read sys.argv.
    """
    parser = argparse.ArgumentParser(description=__doc__.removeprefix("@brief "))
    commands = parser.add_subparsers(dest="command", required=True)
    blender = commands.add_parser("blender", help="Generate train, val and test views with Cycles")
    mujoco = commands.add_parser("mujoco", help="Generate independent regular and stress holdouts")
    preview = commands.add_parser("preview", help="Save one RGB image and its landmark overlay")
    for command in (blender, mujoco, preview):
        command.add_argument("--out", required=True, type=Path, help="New or empty output directory")
        command.add_argument("--width", type=int, default=CameraConfig.width, help="Image width (pixel)")
        command.add_argument("--height", type=int, default=CameraConfig.height, help="Image height (pixel)")
        command.add_argument("--fovy", type=float, default=CameraConfig.fovy, help="Vertical field of view (rad)")
    for command in (blender, preview):
        command.add_argument("--samples", type=int, default=8, help="Blender Cycles samples per pixel")
    blender.add_argument("--train", type=int, default=1024)
    blender.add_argument("--val", type=int, default=96)
    blender.add_argument("--test", type=int, default=64)
    blender.add_argument("--seed", type=int, default=2026111804)
    blender.add_argument("--appearance", choices=["random", "default"], default="random")
    mujoco.add_argument("--regular", type=int, default=160)
    mujoco.add_argument("--stress", type=int, default=64)
    mujoco.add_argument("--seed", type=int, default=2026111802)
    mujoco.add_argument("--exclude", nargs="+", type=Path, default=[], metavar="DIR")
    preview.add_argument("--backend", choices=["blender", "mujoco"], default="mujoco")
    preview.add_argument("--state", choices=list(LightState), default=LightState.READY)
    preview.add_argument("--team", choices=list(Team), default=Team.RED)
    preview.add_argument("--time", type=float, default=0.0, help="Lighting time (s)")
    preview.add_argument("--target-angle", type=float, default=0.0, help="Desired lamp indication about core +X (rad)")
    preview.add_argument("--current-angle", type=float, default=0.0, help="Current lamp indication about core +X (rad)")
    preview.add_argument(
        "--target",
        type=float,
        nargs=3,
        default=Viewpoint.target,
        metavar=("X", "Y", "Z"),
        help="Look-at point in the core frame (m)",
    )
    preview.add_argument("--r", type=float, default=Viewpoint.r, help="Camera radius (m)")
    preview.add_argument("--theta", type=float, default=0.0, help="Camera azimuth (rad)")
    preview.add_argument("--alpha", type=float, default=0.0, help="Camera elevation (rad)")
    preview.add_argument("--roll", type=float, default=0.0, help="Optical-axis rotation (rad)")
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    camera = CameraConfig(args.width, args.height, args.fovy)
    if args.command == "preview":
        view = Viewpoint(args.r, args.theta, args.alpha, tuple(args.target), args.roll)
        require_empty(args.out)
        with TechCoreSimulator(args.backend, camera, args.samples) as simulator:
            frame = simulator.render(view, args.state, args.team, args.time, args.target_angle, args.current_angle)
            Image.fromarray(frame.rgb).save(args.out / "frame.png")
            frame.overlay().save(args.out / "overlay.png")
    elif args.command == "blender":
        generate_dataset(
            args.out,
            "blender",
            {"train": args.train, "val": args.val, "test": args.test},
            args.seed,
            camera,
            argv,
            samples=args.samples,
            appearance=args.appearance,
        )
    else:
        generate_dataset(
            args.out,
            "mujoco",
            {"regular": args.regular, "stress": args.stress},
            args.seed,
            camera,
            argv,
            exclude=args.exclude,
        )


if __name__ == "__main__":
    main()
