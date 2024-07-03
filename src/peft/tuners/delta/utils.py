import torch.nn as nn
import torch
import torch.nn.functional as F



def init_layers_with_active_block(module, active_block, prefix='', root_module=None):
    """
    Recursively traverse all submodules of a model and apply a specific operation
    to the innermost layers, passing the full name of the layer.
    """
    # Initialize root_module on the first call
    if root_module is None:
        root_module = module

    has_children = False
    for name, layer in module.named_children():
        has_children = True
        new_prefix = f"{prefix}.{name}" if prefix else name
        init_layers_with_active_block(layer, active_block, new_prefix, root_module)
    
    # If the module has no children, it's an innermost layer
    if not has_children:
        inner_module = prefix.split(".")[-1]  # like delta_theta, delta_A
        before_module = ".".join(prefix.split(".")[:-1])
        if inner_module == "delta_theta":
            basic_index = 3
            if "base_model" in prefix:
                basic_index += 1
            layer_num = prefix.split(".")[basic_index]
            # this is without the first `model` name
            layer_prefix = ".".join(prefix.split(".")[:basic_index])
            attention_name = ".".join(prefix.split(".")[-basic_index + 1:-1])

            # model.model.layers.0.self_attn.q_proj.base_layer
            # peft: base_model.model.model.layers.0.
            # print(f"root_module.{layer_prefix}[{layer_num}].{attention_name}")
            the_layer = eval(f"root_module.{layer_prefix}[{layer_num}].{attention_name}")
            
            # init that prefix
            if any(p in prefix for p in active_block):
                the_layer.spawn_delta_matrix("default")



def del_and_create_with_active_block(module, active_block, prefix='', root_module=None, collected_values=None):
    """
    Recursively traverse all submodules of a model and apply a specific operation
    to the innermost layers, passing the full name of the layer.
    Collect values returned by del_delta_create_AB and calculate their average.
    """
    # Initialize root_module and collected_values on the first call
    if root_module is None:
        root_module = module
    if collected_values is None:
        collected_values = []

    has_children = False
    for name, layer in module.named_children():
        has_children = True
        new_prefix = f"{prefix}.{name}" if prefix else name
        del_and_create_with_active_block(layer, active_block, new_prefix, root_module, collected_values)
    
    # If the module has no children, it's an innermost layer
    if not has_children:
        inner_module = prefix.split(".")[-1]  # like delta_theta, delta_A
        before_module = ".".join(prefix.split(".")[:-1])
        if (inner_module == "delta_A") or (inner_module == "delta_B"):
            basic_index = 3
            if "base_model" in prefix:
                basic_index += 1
            layer_num = prefix.split(".")[basic_index]
            # this is without the first `model` name
            layer_prefix = ".".join(prefix.split(".")[:basic_index])
            attention_name = ".".join(prefix.split(".")[-basic_index + 1:-1])

            # model.model.layers.0.self_attn.q_proj.base_layer
            the_layer = eval(f"root_module.{layer_prefix}[{layer_num}].{attention_name}")
            
            # init that prefix
            if any(p in prefix for p in active_block):
                value = the_layer.del_delta_create_AB("default")
                collected_values.append(value)

    # On the initial call, return the average of collected values
    if root_module is module:
        if collected_values:
            avg_value = sum(collected_values) / len(collected_values)
            return avg_value
        else:
            return None


def low_rank_proj(delta_theta, r):
    original_dtype = delta_theta.dtype
    delta_theta = delta_theta.to(torch.float32)
    
    U, S, Vh = torch.linalg.svd(delta_theta, full_matrices=True, driver=None)

    # choose first r singular value
    U = U[:, :r]
    S = S[:r]
    U = U @ torch.diag(S)
    Vh = Vh.T[:r, :]

    reconstructed = U @ Vh
    loss = F.mse_loss(delta_theta, reconstructed)

    U = U.to(original_dtype)
    Vh = Vh.to(original_dtype)

    return U, Vh, loss