import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


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
        # A or B, just choose one
        if (inner_module == "delta_A"):
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


def del_delta_create_lion_like(module, active_block, prefix='', root_module=None):
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
        del_delta_create_lion_like(layer, active_block, new_prefix, root_module)
    
    # If the module has no children, it's an innermost layer
    if not has_children:
        inner_module = prefix.split(".")[-1]  # like delta_theta, delta_A
        before_module = ".".join(prefix.split(".")[:-1])
        # if inner_module == "delta_theta":
        if inner_module == "gamma":
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
                the_layer.del_delta_create_gamma_sign("default")


def low_rank_proj(delta_theta, r):
    original_dtype = delta_theta.dtype
    # delta adapter: nn.Linear(1024, 2816)
    # delta_theta shape: torch.Size([2816, 1024])
    delta_theta = delta_theta.to(torch.float32)
    
    # U, S, Vh 's shape
    # (torch.Size([2816, 2816]), torch.Size([1024]), torch.Size([1024, 1024]))
    U, S, Vh = torch.linalg.svd(delta_theta, full_matrices=True, driver=None)

    # choose first r singular value
    U = U[:, :r]
    S = S[:r]
    Vh = Vh[:r, :]

    # U, Vh's shape
    # (torch.Size([2816, 32]), torch.Size([32, 1024]))

    reconstructed = U @ torch.diag(S) @ Vh 
    loss = F.mse_loss(delta_theta, reconstructed)

    U = U.to(original_dtype)
    Vh = Vh.to(original_dtype)
    S = S.to(original_dtype)

    # to calculate: delta_theta_tilde = U @ torch.diag(S) @ Vh 
    # = B @ torch.diag(S) @ A
    # A, B, S, loss
    return Vh, U, S, loss


def simplified_sign(tensor):
        return torch.where(tensor < 0, torch.tensor(-1, dtype=torch.int8, device=tensor.device), torch.tensor(1, dtype=torch.int8, device=tensor.device))

def pack_sign(tensor):
    sign_matrix = simplified_sign(tensor)
    bool_matrix = (sign_matrix > 0).to(torch.uint8).cpu().numpy()
    packed_sign_matrix = np.packbits(bool_matrix)
    stored_data = {
    'packed_sign_matrix': packed_sign_matrix,
    'original_shape': bool_matrix.shape,
    'dtype': tensor.dtype,
    'device': tensor.device
    }
    
    return stored_data

def unpack_and_restore(stored_data):
    packed_sign_matrix = stored_data['packed_sign_matrix']
    original_shape = stored_data['original_shape']
    
    unpacked_bool_matrix = np.unpackbits(packed_sign_matrix)[:np.prod(original_shape)]
    unpacked_bool_matrix = torch.tensor(unpacked_bool_matrix, dtype=torch.uint8).view(original_shape)
    
    data_type = stored_data['dtype']
    device = stored_data['device']
    # back to -1, 1
    unpacked_sign_matrix = torch.where(unpacked_bool_matrix == 1, torch.tensor(1), torch.tensor(-1)).to(data_type).to(device)
    return unpacked_sign_matrix
