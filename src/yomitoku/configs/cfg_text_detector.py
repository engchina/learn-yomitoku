from dataclasses import dataclass, field
from typing import List


@dataclass
class BackBorn:
    name: str = "resnet50"
    dilation: bool = True


@dataclass
class Decoder:
    in_channels: list[int] = field(
        default_factory=lambda: [256, 512, 1024, 2048]
    )
    hidden_dim: int = 256
    adaptive: bool = True
    serial: bool = True
    smooth: bool = False
    k: int = 50


@dataclass
class Data:
    shortest_size: int = 1280
    limit_size: int = 1600


@dataclass
class PostProcess:
    min_size: int = 2
    thresh: float = 0.3
    box_thresh: float = 0.5
    max_candidates: int = 1500
    unclip_ratio: float = 1.8


@dataclass
class Visualize:
    color: List[int] = field(default_factory=lambda: [0, 255, 0])
    heatmap: bool = False


@dataclass
class TextDetectorConfig:
    weights: str = "weights/dbnet_res50_20241111.pth"
    backbone: BackBorn = BackBorn()
    decoder: Decoder = Decoder()
    data: Data = Data()
    post_process: PostProcess = PostProcess()
    visualize: Visualize = Visualize()