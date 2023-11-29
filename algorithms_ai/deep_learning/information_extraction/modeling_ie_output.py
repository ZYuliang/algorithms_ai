from dataclasses import dataclass

import torch
from transformers.utils import ModelOutput
from typing import Optional


@dataclass
class GPOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None

    entity_outputs: Optional[torch.FloatTensor] = None
    relations_outputs: Optional[torch.FloatTensor] = None
    so_head_outputs: Optional[torch.FloatTensor] = None
    so_tail_outputs: Optional[torch.FloatTensor] = None
    input_ids_mask: Optional[torch.FloatTensor] = None
