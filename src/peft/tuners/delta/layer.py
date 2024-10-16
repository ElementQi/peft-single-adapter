import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import svd_lowrank
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_module_weight
from peft.utils.other import transpose

from .utils import low_rank_proj

class DeltaLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("delta_theta", "delta_embedding")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "delta_alpha", "scaling", "delta_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.delta_alpha = {}
        self.scaling = {}
        self.delta_dropout = nn.ModuleDict({})
        self.delta_theta = nn.ModuleDict({})
        # For Low Rank Projection
        self.delta_A = nn.ModuleDict({})
        self.delta_B = nn.ModuleDict({})
        # For Embedding layer
        self.delta_embedding = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self._caches: dict[str, Any] = {}
        self.kwargs = kwargs
        # active means in BAdam, whether a block is active
        # if active, delta theta. Otherwise, delta_A, delta_B
        self.is_active = False

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "EetqLinear":
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self, adapter_name, r, delta_alpha, delta_dropout, init_lora_weights, use_rslora
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.delta_alpha[adapter_name] = delta_alpha
        if delta_dropout > 0.0:
            delta_dropout_layer = nn.Dropout(p=delta_dropout)
        else:
            delta_dropout_layer = nn.Identity()

        self.delta_dropout.update(nn.ModuleDict({adapter_name: delta_dropout_layer}))
        # Actual trainable parameters
        # we should modify this
        # self.bias should also be concerned
        if init_lora_weights:
            self.delta_theta[adapter_name] = nn.Linear(self.in_features, self.out_features, bias=True)


        if use_rslora:
            self.scaling[adapter_name] = delta_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = delta_alpha / r

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        # when do initialization, we have no need to set the adapters to be trainable or not
        self.set_adapter(self.active_adapters)

    def spawn_delta_matrix(self, adapter_name):
        if self.bias == "none":
            self.delta_theta[adapter_name] = nn.Linear(self.in_features, self.out_features, bias=False)
            nn.init.zeros_(self.delta_theta[adapter_name].weight)
        else:
            self.delta_theta[adapter_name] = nn.Linear(self.in_features, self.out_features, bias=True)
            nn.init.zeros_(self.delta_theta[adapter_name].weight)
            nn.init.zeros_(self.delta_theta[adapter_name].bias)
        self._move_adapter_to_device_of_base_layer(adapter_name)


    def del_delta_create_AB(self, adapter_name):
        # print("self.delta_theta[adapter_name].weight.data")
        # print(self.delta_theta[adapter_name].weight.data)

        r = self.r[adapter_name]
        # start low rank proj
        # A, B, difference = low_rank_proj(self.delta_theta[adapter_name].weight.data, self.r)
        # there is a transpose
        A, B, loss = low_rank_proj(self.delta_theta[adapter_name].weight.data.T, r)

        # use_bias = False if self.bias == "none" else True
        use_bias = True if self.base_layer.bias is not None else False

        bias_data = self.delta_theta[adapter_name].bias.data

        # delete delta_theta
        del self.delta_theta[adapter_name]
        self.delta_theta = nn.ModuleDict({})

        # create A, B matrix on model
        self.delta_A[adapter_name] = nn.Linear(self.in_features, r, bias=use_bias)
        self.delta_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)

        self.delta_A[adapter_name].weight.data = A
        self.delta_B[adapter_name].weight.data = B

        if self.delta_A[adapter_name].bias is not None and bias_data is not None:
            self.delta_A[adapter_name].bias.data = bias_data

        self._move_adapter_to_device_of_base_layer(adapter_name)

        # difference = "placeholder"
        # return difference
        
        return loss



    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return 
        if adapter_name in self.delta_theta.keys():
            if init_lora_weights is True:
                nn.init.zeros_(self.delta_theta[adapter_name].weight)
                nn.init.zeros_(self.delta_theta[adapter_name].bias)
        if adapter_name in self.delta_embedding.keys():
            nn.init.zeros_(self.delta_embedding[adapter_name])



class Linear(nn.Module, DeltaLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        delta_alpha: int = 1,
        delta_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        DeltaLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            delta_alpha=delta_alpha,
            delta_dropout=delta_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.delta_theta[adapter].weight.device
        dtype = self.delta_theta[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight = self.delta_theta[adapter].weight

        if cast_to_fp32:
            weight = weight.float()

        output_tensor = transpose(weight, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.delta_theta[adapter].weight.data = weight.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # self._check_forward_args(x, *args, **kwargs)

        if self.is_active:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.delta_theta.keys():
                    continue
                delta_theta = self.delta_theta[active_adapter]
                dropout = self.delta_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(delta_theta.weight.dtype)

                result = result + delta_theta(dropout(x)) * scaling
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "delta." + rep