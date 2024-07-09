from marshmallow_dataclass import dataclass


@dataclass
class InferenceConfig:
    with_normals: bool = False
    occ: bool = True
    padding: float = 0.1
    threshold: float = 0.7
    resolution0: int = 32
    upsampling_steps: int = 2
    refinement_steps: int = 0
    vis: bool = False

