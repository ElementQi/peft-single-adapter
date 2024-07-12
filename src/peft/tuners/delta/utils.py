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


def sparse_prune_layers_with_active_block(module, active_block, prefix='', root_module=None):
    """
    Recursively traverse all submodules of a model and apply a specific operation
    to the innermost layers, passing the full name of the layer.
    """
    # Initialize root_module on the first call
    if root_module is None:
        root_module = module

    # Collect layers to process
    layers_to_process = []

    def collect_layers(module, active_block, prefix='', root_module=None):
        has_children = False
        for name, layer in module.named_children():
            has_children = True
            new_prefix = f"{prefix}.{name}" if prefix else name
            collect_layers(layer, active_block, new_prefix, root_module)
        
        # If the module has no children, it's an innermost layer
        if not has_children:
            inner_module = prefix.split(".")[-1]  # like delta_theta, delta_A
            before_module = ".".join(prefix.split(".")[:-1])
            if inner_module == "delta_theta_pruned":
                basic_index = 3
                if "base_model" in prefix:
                    basic_index += 1
                layer_num = prefix.split(".")[basic_index]
                # this is without the first `model` name
                layer_prefix = ".".join(prefix.split(".")[:basic_index])
                attention_name = ".".join(prefix.split(".")[-basic_index + 1:-1])

                the_layer = eval(f"root_module.{layer_prefix}[{layer_num}].{attention_name}")
                
                # init that prefix
                if any(p in prefix for p in active_block):
                    layers_to_process.append(the_layer)

    collect_layers(module, active_block, prefix, root_module)

    # Process layers after collecting
    for layer in layers_to_process:
        layer.del_delta_create_sparse("default")



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


def low_rank_proj_(delta_theta, r):
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
    difference = delta_theta - reconstructed
    for_loss = torch.norm(difference, 'fro') / torch.norm(delta_theta, 'fro')

    U = U.to(original_dtype)
    Vh = Vh.to(original_dtype)
    S = S.to(original_dtype)

    sparse_weight = prune_sparse(difference)

    # to calculate: delta_theta_tilde = U @ torch.diag(S) @ Vh 
    # = B @ torch.diag(S) @ A
    # A, B, S, difference_sparse, loss
    return Vh, U, S, sparse_weight, for_loss


def prune_sparse(difference, pruning_percentage=0.05, bin_num=1000):
    # delta adapter: nn.Linear(1024, 2816)
    # delta_theta shape: torch.Size([2816, 1024])

    difference_32 = difference.float()

    # prune delta using histogram method
    abs_difference = difference_32.abs()
    hist = torch.histc(abs_difference, bins=bin_num)
    cumulative_hist = torch.cumsum(hist, dim=0)
    total_elements = abs_difference.numel()
    threshold_bin = torch.searchsorted(cumulative_hist, (1 - pruning_percentage) * total_elements)
    threshold = (threshold_bin / bin_num) * abs_difference.max()

    # prune delta
    difference_32[difference_32.abs() < threshold] = 0

    # sparse matrix
    sparse_weight = difference_32.to_sparse()

    return sparse_weight