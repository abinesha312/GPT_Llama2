import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and CUDA installed.")
else:
    raise RuntimeError("Running and it is available.")