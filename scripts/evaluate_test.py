import argparse
from inference_multidiffusion import inference
from experiments.config import EXPERIMENTS
from experiments.healpix_dataset import make_dataset_wrapper


def parse_known():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS.keys())
    return parser.parse_known_args()


if __name__ == "__main__":
    args, remaining = parse_known()
    base_wrapper = make_dataset_wrapper(EXPERIMENTS[args.experiment])
    def dataset_wrapper(*, split="test"):
        return base_wrapper(split="test", final_eval=True)
    inference(arg_list=remaining, customized_dataset=dataset_wrapper)