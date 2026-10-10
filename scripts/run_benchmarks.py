"""Run the final learned-model benchmark across models and seeds."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--models', nargs='+', choices=['windno', 'convlstm', 'unet', 'plain_fno'],
                        default=['windno', 'convlstm', 'unet', 'plain_fno'])
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    root = Path(__file__).resolve().parents[1]
    dataset = root / 'datasets/global_airpollution_2015_2024_4d.npz'
    if not dataset.is_file():
        parser.error('Full training tensor is missing. See docs/data-preparation.md.')
    for model in args.models:
        for seed in args.seeds:
            subprocess.run([sys.executable, str(root / 'ML_3/experiments.py'),
                            '--run-training', '--model', model, '--seed', str(seed),
                            '--epochs', str(args.epochs)], cwd=root, check=True)


if __name__ == '__main__':
    main()
