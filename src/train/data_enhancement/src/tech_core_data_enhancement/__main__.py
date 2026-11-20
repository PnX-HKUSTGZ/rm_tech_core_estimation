"""@brief 生成增强训练图像与程序化背景负样本。"""

import argparse
from pathlib import Path

from .offline import run


def main(argv: list[str] | None = None) -> None:
    """@brief 按命令行参数生成增强数据集。

    @param[in] argv 模块名之后的参数；为 None 时读取 sys.argv。
    """
    parser = argparse.ArgumentParser(description=__doc__.removeprefix("@brief "))
    parser.add_argument("--data", required=True, type=Path, help="Input dataset directory")
    parser.add_argument("--out", required=True, type=Path, help="New or empty output directory")
    parser.add_argument("--copies", type=int, default=1, help="Extra variants per training target with standard policy")
    parser.add_argument("--negatives", type=float, default=0.05, help="Backgrounds per output training target")
    parser.add_argument("--policy", choices=["standard", "none"], default="standard")
    parser.add_argument("--seed", type=int, default=20261118)
    run(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
