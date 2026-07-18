import argparse
from cbottle.training import super_resolution
from experiments.config import EXPERIMENTS
from experiments.healpix_dataset import make_dataset_wrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS.keys())
    parser.add_argument(
        "--output-path", type=str, required=True, help="output directory"
    )
    parser.add_argument(
        "--log-freq", type=int, default=100, help="Log every N steps (default: 100)"
    )
    parser.add_argument(
        "--lr-level", type=int, default=6, help="HPX level of the low-resolution map"
    )
    parser.add_argument(
        "--train-batch-size", type=int, default=15, help="training batch size per GPU"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=30, help="validation batch size per GPU"
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=3,
        help="number of workers for training dataloader",
    )
    parser.add_argument(
        "--bf16", action="store_true", help="use bfloat16 precision"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_wrapper = make_dataset_wrapper(EXPERIMENTS[args.experiment])
    super_resolution.train(
        output_path=args.output_path,
        customized_dataset=dataset_wrapper,
        num_steps=int(1e6),
        log_freq=args.log_freq,
        lr_level=args.lr_level,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=args.bf16,
    )