



from transformers import PretrainedConfig
from typing import List

from dataclasses import dataclass, field

@dataclass
class Task:
    task_name:str
    loss_weight:float
    input_column:str
    label_column:str



class MultiTaskConfig(PretrainedConfig):
    model_type = "multi_task"

    def __init__(
        self,
        block_type="bottleneck",
        layers: List[int] = [3, 4, 6, 3],
        num_classes: int = 1000,
        input_channels: int = 3,
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        avg_down: bool = False,
        tasks=('task1', 'task2'),
        encoder ='bert1',
        **kwargs,
    ):
        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(f"`block_type` must be 'basic' or bottleneck', got {block_type}.")
        if stem_type not in ["", "deep", "deep-tiered"]:
            raise ValueError(f"`stem_type` must be '', 'deep' or 'deep-tiered', got {stem_type}.")

        self.block_type = block_type
        self.layers = layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.cardinality = cardinality
        self.base_width = base_width
        self.stem_width = stem_width
        self.stem_type = stem_type
        self.avg_down = avg_down
        self.tasks = tasks
        self.encoder = encoder
        super().__init__(**kwargs)

        # max_seq_length: int = 64
        # embedders: Dict[str, EmbedderConfig] = field(default_factory=dict)
        # tasks: Dict[str, TaskConfig] = field(default_factory=dict)
        # ex_features: Dict[str, ExFeatureConfig] = field(default_factory=dict)
        # adapters: Dict[str, AdapterConfig] = field(default_factory=dict)
        # train_config: TrainConfig = TrainConfig()




# c = MultiTaskConfig()
# print(c)
