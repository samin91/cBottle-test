from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    name: str
    root: str
    fields: list[str]
    mean: dict[str, float]
    scale: dict[str, float]
    train_dir: str = "target_train"
    val_dir: str = "target_validation"   # aliased as "test" during training
    real_test_dir: str = "target_test"   # only touched by evaluate_test.py
    grid_level: int = 7
    chunk_size_train: int = 48
    chunk_size_val: int = 8


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "experiment1": ExperimentConfig(
        name="experiment1",
        root="/work/bk1444/climbench/data/experiment1/healpix",
        fields=["tas"],
        mean={"tas": 286.9937438964844},
        scale={"tas": 15.424193382263184},
    ),
    "experiment2": ExperimentConfig(
        name="experiment2",
        root="/work/bk1444/climbench/data/experiment2/healpix",  # TODO: confirm actual path
        fields=["pt"],  # corrected from table's "pr" — verified against real data
        mean={"pt": None},   # TODO: compute — see script below
        scale={"pt": None},  # TODO: compute
    ),
    "experiment3": ExperimentConfig(
        name="experiment3",
        root="/work/bk1444/climbench/data/experiment3/healpix",  # TODO: confirm actual path
        fields=["msl", "q850", "tas", "tcc", "tp", "u850", "v850", "z500"],
        mean={v: None for v in ["tp", "2t", "hus", "tcc", "z500", "v850", "u850", "msl"]},   # TODO
        scale={v: None for v in ["tp", "2t", "hus", "tcc", "z500", "v850", "u850", "msl"]},  # TODO
    ),
}