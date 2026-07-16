import numpy as np
import torch.distributed as dist
import xarray as xr 
import cbottle.datasets.zarr_loader as zl
import cbottle.datasets.merged_dataset as md
from earth2grid import healpix
import pdb

ROOT = "/work/bk1444/climbench/data/experiment1/healpix"
FIELDS = ["tas"]
MEAN = {"tas": 286.9937438964844}    # <-- replace with your Step 0 numbers
SCALE = {"tas": 15.424193382263184}    # <--
NPIX_HPX7 = 12 * 128**2  # 196608

# cBottle's "test" split is used as validation during training
SPLIT_DIRS = {"train": "target_train", "test": "target_validation"}


def encode_task(times, frames):
    data = frames[0]  # dict {(varname, level): array}; level -1 means 2D
    target = np.stack(
        [(data[(v, zl.NO_LEVEL)] - MEAN[v]) / SCALE[v] for v in FIELDS]
    ).astype(np.float32)[:, None, :]  # shape (channels, 1, npix)
    assert target.shape[-1] == NPIX_HPX7, f"unexpected grid: {target.shape}"
    return {"target": target}


def dataset_wrapper(*, split: str = "train"):
    loader = zl.ZarrLoader(
        path=f"{ROOT}/{SPLIT_DIRS[split]}.zarr",
        variables_2d=FIELDS,
        variables_3d=[],
        levels=[],
    )
    # New data stores time as numeric + calendar="proleptic_gregorian", which
    # ZarrLoader decodes into a CFTimeIndex of cftime objects. merged_dataset.py's
    # pd.Timestamp(time) can't consume those directly, so convert here.
    if isinstance(loader.times, xr.CFTimeIndex):
        loader.times = loader.times.to_datetimeindex()

    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    ds = md.TimeMergedDataset(
        loader.times,
        time_loaders=[loader],
        transform=encode_task,
        chunk_size=48 if split == "train" else 8,
        shuffle=True,
        rank=rank,          # each GPU gets its own share of the 60 years
        world_size=world,
    )
    # metadata the trainer reads to size the network
    ds.grid = healpix.Grid(level=7, pixel_order=healpix.PixelOrder.NEST)
    ds.fields_out = FIELDS
    return ds