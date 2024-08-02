import bitsandbytes as bnb
import torch

from peft.import_utils import is_bnb_4bit_available
from peft.tuners.tuners_utils import BaseTunerLayer
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
            if active_adapter not in self.delta_A.keys():
                continue
            delta_A = self.delta_A[active_adapter]
            delta_B = self.delta_B[active_adapter]
            delta_S = self.delta_S[active_adapter]
            dropout = self.delta_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(delta_A.weight.dtype)

            use_bias = True if delta_B.bias is not None else False

            # delta_tilde = B @ torch.diag(S) @ A
            # output = x @ delta_tilde.weight.T * scaling
            # U, Vh's shape = B, A's shape
            # (torch.Size([2816, 32]), torch.Size([32, 1024]))

            # for precision's concern, using Einstein summation convention
            A = delta_A.weight.T
            B = delta_B.weight.T
            S = delta_S
            bias = delta_B.bias

            # x @ A @ diag(S) @ B + bias
            if use_bias:
                output = (x @ A @ torch.diag(S) @ B + bias) * scaling
                # output = (torch.einsum('ij,jk,k,kl->il', x, A, S, B) + bias) * scaling
            else:
                output = x @ A @ torch.diag(S) @ B * scaling

            # this means delta_theta is not empty
            if active_adapter in self.delta_theta.keys():
                delta_theta = self.delta_theta[active_adapter]
                dropout = self.delta_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    x = x.to(delta_theta.weight.dtype)

                output = delta_theta(dropout(x)) * scaling + output

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
