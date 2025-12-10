import torch
import torch.nn.functional as F
from e3nn import o3
from pathlib import Path
import os

model_path = Path("../models") / f"propensity_model_20251014_batchnorm.pt"

model = torch.load(model_path, weights_only=True, map_location="cpu")

for n in range(4):
    print("For Layer Number ", n+1)
    print(" - Batch Norm Weights \n", model[f"interactions.{n}.batch_norm_up.weight"])
    print(" - Batch Norm Biases \n", model[f"interactions.{n}.batch_norm_up.bias"])
    print(" - Batch Norm Running Mean \n", model[f"interactions.{n}.batch_norm_up.running_mean"])
    print(" - Batch Norm Running Var \n", model[f"interactions.{n}.batch_norm_up.running_var"])