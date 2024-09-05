import torch

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, '..', 'functional')
sys.path.append(module_path)

from quantization import (
    quantize_weight_per_channel_absmax,
    dynamic_quantize_activation_per_token_absmax,
    dequantize_activation_w_per_channel_a_per_token,
)

import cupy

# NOTE(jpyo0803): dynamic activaiton per token / static weight per channel
class CustomW8A8BFP32OFP32Linear(torch.nn.Module):
    # For fc2 and out_proj
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('weight_scales', torch.ones(
            self.out_features, dtype=torch.float16, requires_grad=False))

    @torch.no_grad()
    def forward(self, x):
        assert x.dtype == torch.float16
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])

        int8_x, x_scales = dynamic_quantize_activation_per_token_absmax(x.clone().detach())

        x_cupy = cupy.from_dlpack(int8_x.to(torch.int32))
        weight_T_cupy = cupy.from_dlpack(self.weight.transpose(-2, -1).to(torch.int32))

        y_cupy = cupy.matmul(x_cupy, weight_T_cupy)

        y = torch.from_dlpack(y_cupy)
        y = dequantize_activation_w_per_channel_a_per_token(y, self.weight_scales, x_scales)

        y = y.to(torch.float16)
        y = y.view(*x_shape[:-1], -1)

        return y

    @staticmethod
    def from_float(module: torch.nn.Linear):
        
        int8_module = CustomW8A8BFP32OFP32Linear(
            module.in_features, module.out_features)

        int8_module.weight, weight_scales = quantize_weight_per_channel_absmax(module.weight)
        int8_module.weight_scales = weight_scales.squeeze()


        return int8_module