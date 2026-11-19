"""@brief Train, evaluate and export the technology core RGB landmark models."""

import argparse
from pathlib import Path

import torch


def main(argv=None):
    """@brief Dispatch training workspace commands; argv defaults to the process arguments.

    @param[in] argv Command-line arguments, or None for process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.removeprefix("@brief "))
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("detect", "pose", "keypoints", "finetune", "evaluate", "export"):
        p = commands.add_parser(command)
        p.add_argument("--out", required=True, type=Path, help="Empty output directory")
        if command != "export":
            p.add_argument("--data", required=True, type=Path, help="Manifest directory")
            p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Torch device")
        if command in ("detect", "pose", "keypoints", "finetune"):
            p.add_argument("--seed", type=int, default=20261118)
            p.add_argument("--epochs", type=int, default={"keypoints": 40, "finetune": 30}.get(command, 20))
            p.add_argument("--batch", type=int, default=8 if command in ("pose", "detect") else 16)
        if command in ("detect", "pose"):
            p.add_argument("--size", choices=list("nsmlx"), default="n")
            p.add_argument("--imgsz", type=int, default=320, help="Square input side (pixel)")
        if command in ("keypoints", "finetune"):
            p.add_argument("--learning-rate", type=float, default=0.001 if command == "keypoints" else 0.0001)
        if command == "keypoints":
            p.add_argument("--variant", type=int, choices=(18, 30), default=18)
        if command == "finetune":
            for name in ("initial", "dev-data", "detector"):
                p.add_argument("--" + name, required=True, type=Path)
        if command == "evaluate":
            route = p.add_mutually_exclusive_group(required=True)
            route.add_argument("--detector", type=Path, help="Two-stage detector run or checkpoint")
            route.add_argument("--pose", type=Path, help="Direct pose run or checkpoint")
            p.add_argument("--keypoints", type=Path, help="LiteHRNet checkpoint paired with --detector")
            selection = p.add_mutually_exclusive_group()
            selection.add_argument("--keypoint-threshold", type=float, help="Fixed threshold declared before inference")
            selection.add_argument("--selection", type=Path, help="Use a frozen selection.yaml")
            selection.add_argument("--dev-data", type=Path, help="Calibrate on separate MuJoCo development data")
        if command == "export":
            for name in ("detector", "keypoints", "evaluation"):
                p.add_argument("--" + name, required=True, type=Path)
            p.add_argument("--verify", action="store_true", help="Replay golden images with ONNX Runtime")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    if command in ("detect", "pose"):
        from .yolo import train_yolo

        train_yolo(task=command, **args)
    elif command == "keypoints":
        from .litehrnet.train import train

        train(**args)
    elif command == "finetune":
        from .litehrnet.train import finetune

        finetune(**args)
    elif command == "evaluate":
        from .evaluate import evaluate

        if bool(args["detector"]) != bool(args["keypoints"]):
            parser.error("--detector and --keypoints must be supplied together")
        evaluate(**args)
    else:
        from .export import export

        export(**args)


if __name__ == "__main__":
    main()
