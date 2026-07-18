from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    name: str
    root: str
    fields: list[str]
    mean: dict[str, float]
    scale: dict[str, float]
    train_dir: str = "target_train"       # only target is needed. Input is computed from target. 
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
        mean={"tas": 286.99915542783003}, 
        scale={"tas": 15.414200571501576},
    ),
    "experiment2": ExperimentConfig(
        name="experiment2",
        root="/work/bk1444/climbench/data/experiment2/healpix",  
        fields=["tp"],  
        mean={"tp": 0.0028682559418698973},  
        scale={"tp": 0.006175653414683115},  
    ),
    "experiment3": ExperimentConfig(
        name="experiment3",
        root="/work/bk1444/climbench/data/experiment3/healpix",  
        fields=["msl", "q850", "tas", "tcc", "tp", "u850", "v850", "z500"],
        mean={
            "msl": 101139.39094338276,
            "q850": 0.005964962388804392,
            "tas": 286.99915542783003,
            "tcc": 0.6229747817337858,
            "tp": 0.0028682559418698973,
            "u850": 1.1146329207266692,
            "v850": 0.09364891663880362,
            "z500": 55433.52487005346,
        },
        scale={
            "msl": 1062.9293798393428,
            "q850": 0.004028018112375323,
            "tas": 15.414200571501576,
            "tcc": 0.3009191133842002,
            "tp": 0.006175653414683115,
            "u850": 7.60048045817027,
            "v850": 5.052339021547254,
            "z500": 2704.6219155593317,
        },
        
    ),
}