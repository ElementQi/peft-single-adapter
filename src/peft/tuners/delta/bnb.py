
from typing import Any, Optional

import bitsandbytes as bnb
import torch

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_bnb_weight
from peft.utils.other import transpose

from .layer import DeltaLayer


class Linear4bit(torch.nn.Module, DeltaLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            r: int = 0,
            delta_alpha: int = 1,
            delta_dropout: float = 0.0,
            init_lora_weights: bool = True,
            use_rslora: bool = False,
            **kwargs,
        ) -> None:
            super().__init__()
            DeltaLayer.__init__(self, base_layer)
            self.fan_in_fan_out = False

            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                r,
                delta_alpha=delta_alpha,
                delta_dropout=delta_dropout,
                init_lora_weights=init_lora_weights,
                use_rslora=use_rslora,
            )


        def get_delta_weight(self, adapter):
            return (
                transpose(
                    self.delta_theta[adapter].weight,
                    False,
                )
                * self.scaling[adapter]
            )


        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            # self._check_forward_args(x, *args, **kwargs)
            adapter_names = kwargs.pop("adapter_names", None)

            result = self.base_layer(x, *args, **kwargs)
            # As per Tim Dettmers, for 4bit, we need to defensively clone here.
            # The reason is that in some cases, an error can occur that backprop
            # does not work on a manipulated view. This issue may be solved with
            # newer PyTorch versions but this would need extensive testing to be
            # sure.
            result = result.clone()

            for active_adapter in self.active_adapters:
                # if active_adapter not in self.lora_A.keys():
                #     continue

                # this means delta_theta is empty
                
                if active_adapter not in self.delta_theta.keys():
                    if active_adapter not in self.delta_A.keys():
                        continue
                    else:
                        delta_A = self.delta_A[active_adapter]
                        delta_B = self.delta_B[active_adapter]
                        dropout = self.delta_dropout[active_adapter]
                        scaling = self.scaling[active_adapter]

                        requires_conversion = not torch.is_autocast_enabled()
                        if requires_conversion:
                            expected_dtype = result.dtype
                            x = x.to(delta_A.weight.dtype)
                            # x = x.to(self.base_layer[active_adapter].weight.dtype)
                        
                        # print(f" x[0] shape: {x[0].shape}")
                        # print(f" A shape: {delta_A.weight.shape}")
                        # print(f"delta_A.in_features: {delta_A.in_features}")
                        # print(f"delta_A.out_features: {delta_A.out_features}")
                        # print(f"delta_A.weight: {delta_A.weight}")
                        # print(f"delta_A.bias: {delta_A.bias}")


                        # batch_size, seq_len, in_features = x.shape
                        # x = x.view(batch_size * seq_len, in_features)  # [batch_size * seq_len, in_features]

                        # output_1 = delta_A(dropout(x))  # [batch_size * seq_len, out_features]
                        # output_1 = output_1.view(batch_size, seq_len, -1)  # [batch_size, seq_len, out_features]

                        # # Apply delta_B
                        # output = delta_B(output_1) * scaling

                        # output_1 = delta_A(dropout(x))
                        # output = delta_B(output_1) * scaling

                        # TODO: make sure bias here works well
                        use_bias = True if delta_A.bias is not None else False
                        W_A = delta_A.weight.T
                        W_B = delta_B.weight.T
                        if use_bias:
                            # output = (dropout(x) @ delta_A.weight @ delta_B.weight + delta_A.bias)  * scaling
                            output = (dropout(x) @ W_A @ W_B + delta_A.bias)  * scaling
                        else:
                            # output = dropout(x) @ delta_A.weight @ delta_B.weight * scaling
                            output = dropout(x) @ W_A @ W_B * scaling
                        # output = delta_B(delta_A(dropout(x))) * scaling
                        # output = dropout(x) @ 


                        
                else:
                    delta_theta = self.delta_theta[active_adapter]
                    dropout = self.delta_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        x = x.to(delta_theta.weight.dtype)

                    output = delta_theta(dropout(x)) * scaling
                        
                if requires_conversion:
                    output = output.to(expected_dtype)

                result = result + output

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "delta." + rep

def dispatch_bnb_4bit(target: torch.nn.Module, adapter_name: str, **kwargs):
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    loaded_in_4bit = kwargs.get("loaded_in_4bit", False)
    if loaded_in_4bit and is_bnb_4bit_available() and isinstance(target_base_layer, bnb.nn.Linear4bit):
        fourbit_kwargs = kwargs.copy()
        fourbit_kwargs.update(
            {
                "compute_dtype": target_base_layer.compute_dtype,
                "compress_statistics": target_base_layer.weight.compress_statistics,
                "quant_type": target_base_layer.weight.quant_type,
            }
        )
        new_module = Linear4bit(target, adapter_name, **fourbit_kwargs)

    return new_module