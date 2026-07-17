import torch.distributed as dist
import xarray as xr
import numpy as np
import cbottle.datasets.zarr_loader as zl
import cbottle.datasets.merged_dataset as md
from earth2grid import healpix

from .config import ExperimentConfig


def make_dataset_wrapper(config: ExperimentConfig):
    """Returns a dataset_wrapper closure configured for one experiment.

    Training use: dataset_wrapper(split="train") / dataset_wrapper(split="test")
        -> "test" is aliased to the VALIDATION set, matching current training behavior.
    Final eval use: dataset_wrapper(split="test", final_eval=True)
        -> "test" now means the real held-out test set. Only call this once,
           after the checkpoint is already chosen.
    """
    split_dirs = {"train": config.train_dir, "test": config.val_dir}
    chunk_sizes = {"train": config.chunk_size_train, "test": config.chunk_size_val}

    def encode_task(times, frames):
        data = frames[0]
        target = np.stack(
            [(data[(v, zl.NO_LEVEL)] - config.mean[v]) / config.scale[v] for v in config.fields]
        ).astype(np.float32)[:, None, :]
        npix = 12 * 4**config.grid_level
        assert target.shape[-1] == npix, f"unexpected grid: {target.shape}"
        return {"target": target}

    def dataset_wrapper(*, split: str = "train", final_eval: bool = False):
        split_dir = config.real_test_dir if final_eval else split_dirs[split]

        loader = zl.ZarrLoader(
            path=f"{config.root}/{split_dir}.zarr",
            variables_2d=config.fields, variables_3d=[], levels=[],
        )
        if isinstance(loader.times, xr.CFTimeIndex):
            loader.times = loader.times.to_datetimeindex()

        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        ds = md.TimeMergedDataset(
            loader.times, time_loaders=[loader], transform=encode_task,
            chunk_size=chunk_sizes.get(split, 8),
            shuffle=not final_eval,   # deterministic single pass for final eval
            rank=rank, world_size=world,
        )
        ds.grid = healpix.Grid(level=config.grid_level, pixel_order=healpix.PixelOrder.NEST)
        ds.fields_out = config.fields
        return ds

    return dataset_wrapper