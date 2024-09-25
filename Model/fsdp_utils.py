import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def fsdp_wrap_policy(model):
    """
    Define the auto-wrap policy for FSDP based on model size.
    """
    return size_based_auto_wrap_policy(model, min_num_params=1e8)

def fsdp_mixed_precision():
    """
    Define mixed precision for FSDP, using bfloat16 for faster computation.
    """
    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

def create_fsdp_model(model):
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=size_based_auto_wrap_policy,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
        device_id=torch.cuda.current_device()
    )
    return fsdp_model