from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math
import os
import json
from typing import Dict, Any
from datetime import datetime, timezone


def name_with_datetime():
    now = datetime.now(tz=timezone.utc)
    return now.strftime("%Y-%m-%d_%H:%M")

def load_json_file(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)

def save_json_file(data: Dict[str, Any], path: str) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def get_model_name_from_path(path: str) -> str:
    return os.path.basename(path)

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)