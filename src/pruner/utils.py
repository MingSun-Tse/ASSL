import torch, torch.nn as nn
from collections import OrderedDict
from fnmatch import fnmatch, fnmatchcase
import math, numpy as np, copy
tensor2list = lambda x: x.data.cpu().numpy().tolist()
tensor2array = lambda x: x.data.cpu().numpy()
totensor = lambda x: torch.Tensor(x)

def get_pr_layer(base_pr, layer_name, layer_index, skip=[], compare_mode='local'):
    """ 'base_pr' example: '[0-4:0.5, 5:0.6, 8-10:0.2]', 6, 7 not mentioned, default value is 0
    """
    if compare_mode in ['global']:
        pr = 1e-20 # a small positive value to indicate this layer will be considered for pruning, will be replaced
    elif compare_mode in ['local']:
        pr = base_pr[layer_index]

    # if layer name matchs the pattern pre-specified in 'skip', skip it (i.e., pr = 0)
    for p in skip:
        if fnmatch(layer_name, p):
            pr = 0
    return pr

def get_pr_model(layers, base_pr, skip=[], compare_mode='local'):
    """Get layer-wise pruning ratio for a model
    """
    pr = OrderedDict()
    if base_pr: # 'stage_pr' may be None (in the case that 'base_pr_model' is provided)
        if compare_mode in ['global']:
            pr['model'] = base_pr
        for name, layer in layers.items():
            pr[name] = get_pr_layer(base_pr, name, layer.index, skip=skip, compare_mode=compare_mode)
    else:
        raise NotImplementedError
    return pr

def get_constrained_layers(layers, constrained_layers_pattern):
    constrained_layers = []
    for name, _ in layers.items():
        for p in constrained_layers_pattern:
            if fnmatch(name, p):
                constrained_layers += [name]
    return constrained_layers

def adjust_pr(layers, pr, pruned, kept, num_pruned_constrained, constrained_layers):
    """The real pr of a layer may not be exactly equal to the assigned one due to various reasons (e.g., constrained layers). 
    Adjust it here to make them exact the same.
    """
    pr, pruned, kept = copy.deepcopy(pr), copy.deepcopy(pruned), copy.deepcopy(kept)
    for name, layer in layers.items():
        if name in constrained_layers:
            # -- averaging within all constrained layers to keep the total num of pruned weight groups still the same
            num_pruned = int(num_pruned_constrained / len(constrained_layers))
            # --
            pr[name] = num_pruned / len(layer.score)
            order = pruned[name] + kept[name]
            pruned[name], kept[name] = order[:num_pruned], order[num_pruned:]
        else:
            num_pruned = len(pruned[name])
            pr[name] = num_pruned / len(layer.score)
    return pr, pruned, kept

def get_score_layer(module, wg='filter', criterion='l1-norm'):
    r"""Get importance score for a layer.

    Return:
        A dict that has key 'score', whose value is a list
    """
    # -- define any scoring scheme here as you like
    shape = module.weight.data.shape
    if wg == "channel":
        l1 = module.weight.abs().mean(dim=[0, 2, 3]) if len(shape) == 4 else module.weight.abs().mean(dim=0)
    elif wg == "filter":
        l1 = module.weight.abs().mean(dim=[1, 2, 3]) if len(shape) == 4 else module.weight.abs().mean(dim=1)
    elif wg == "weight":
        l1 = module.weight.abs().flatten()
    # --

    out = {}
    out['l1-norm'] = tensor2list(l1)
    if hasattr(module, 'wn_scale'): 
        out['wn_scale'] = tensor2list(module.wn_scale.abs())
    out['score'] = out[criterion]
    return out

def pick_pruned_layer(score, pr=None, threshold=None, sort_mode='min'):
    r"""Get the indices of pruned weight groups in a layer.
    """
    assert sort_mode in ['min', 'rand', 'max']
    score = np.array(score)
    num_total = len(score)
    if sort_mode in ['rand']:
        assert pr is not None
        num_pruned = min(math.ceil(pr * num_total), num_total - 1) # do not prune all
        order = np.random.permutation(num_total).tolist()
    else:
        num_pruned = math.ceil(pr * num_total) if threshold is None else len(np.where(score < threshold)[0])
        num_pruned = min(num_pruned, num_total - 1) # do not prune all
        if sort_mode in ['min', 'ascending']:
            order = np.argsort(score).tolist()
        elif sort_mode in ['max', 'descending']:
            order = np.argsort(score)[::-1].tolist()
    pruned, kept = order[:num_pruned], order[num_pruned:]
    return pruned, kept

def pick_pruned_model(layers, pr, wg='filter', criterion='l1-norm', compare_mode='local', sort_mode='min', constrained=[]):
    r"""Pick pruned weight groups for a model.
    Args:
        layers: an OrderedDict, key is layer name

    Return:
        pruned (OrderedDict): key is layer name, value is the pruned indices for the layer
        kept (OrderedDict): key is layer name, value is the kept indices for the layer
    """
    assert sort_mode in ['rand', 'min', 'max'] and compare_mode in ['global', 'local']
    pruned, kept = OrderedDict(), OrderedDict()
    all_scores, num_pruned_constrained = [], 0

    # iter to get importance score for each layer
    for name, layer in layers.items():
        out = get_score_layer(layer.module, wg=wg, criterion=criterion)
        score = out['score']
        layer.score = score
        if pr[name] > 0: # pr > 0 indicates we want to prune this layer so its score will be included in the <all_scores>
            all_scores += score

        # local pruning
        if compare_mode in ['local']:
            assert isinstance(pr, dict)
            pruned[name], kept[name] = pick_pruned_layer(score, pr[name], sort_mode=sort_mode)
            if name in constrained: 
                num_pruned_constrained += len(pruned[name])
    
    # global pruning
    if compare_mode in ['global']:
        num_total = len(all_scores)
        num_pruned = min(math.ceil(pr['model'] * num_total), num_total - 1) # do not prune all
        if sort_mode == 'min':
            threshold = sorted(all_scores)[num_pruned] # in ascending order
        elif sort_mode == 'max':
            threshold = sorted(all_scores)[::-1][num_pruned] # in decending order
        print(f'#all_scores: {len(all_scores)} threshold:{threshold:.6f}')

        for name, layer in layers.items():
            if pr[name] > 0:
                if sort_mode in ['rand']:
                    pass
                elif sort_mode in ['min', 'max']:
                    pruned[name], kept[name] = pick_pruned_layer(layer.score, pr=None, threshold=threshold, sort_mode=sort_mode)
            else:
                pruned[name], kept[name] = [], list(range(len(layer.score)))
            if name in constrained: 
                num_pruned_constrained += len(pruned[name])
    
    # adjust pr/pruned/kept
    pr, pruned, kept = adjust_pr(layers, pr, pruned, kept, num_pruned_constrained, constrained)
    print(f'Adjust pr/pruned/kept done.')
    
    return pr, pruned, kept

def get_next_learnable(layers, layer_name, n_conv_within_block=3):
    r"""Get the next learnable layer for the layer of 'layer_name', chosen from 'layers'.
    """
    current_layer = layers[layer_name]

    # for standard ResNets on ImageNet
    if hasattr(current_layer, 'block_index'):
        block_index = current_layer.block_index
        if block_index == n_conv_within_block - 1:
            return None
    
    for name, layer in layers.items():
        if layer.type == current_layer.type and layer.index == current_layer.index + 1:
            return name
    return None

def get_prev_learnable(layers, layer_name):
    r"""Get the previous learnable layer for the layer of 'layer_name', chosen from 'layers'.
    """
    current_layer = layers[layer_name]

    # for standard ResNets on ImageNet
    if hasattr(current_layer, 'block_index'):
        block_index = current_layer.block_index
        if block_index in [None, 0, -1]: # 1st conv, 1st conv in a block, 1x1 shortcut layer
            return None
    
    for name, layer in layers.items():
        if layer.index == current_layer.index - 1:
            return name
    return None

def get_next_bn(model, layer_name):
    r"""Get the next bn layer for the layer of 'layer_name', chosen from 'model'.
    Return the bn module instead of its name.
    """
    just_passed = False
    for name, module in model.named_modules():
        if name == layer_name:
            just_passed = True
        if just_passed and isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            return module
    return None

def replace_module(model, name, new_m):
    """Replace the module <name> in <model> with <new_m>
    E.g., 'module.layer1.0.conv1' ==> model.__getattr__('module').__getattr__("layer1").__getitem__(0).__setattr__('conv1', new_m)
    """
    obj = model
    segs = name.split(".")
    for ix in range(len(segs)):
        s = segs[ix]
        if ix == len(segs) - 1: # the last one
            if s.isdigit():
                obj.__setitem__(int(s), new_m)
            else:
                obj.__setattr__(s, new_m)
            return
        if s.isdigit():
            obj = obj.__getitem__(int(s))
        else:
            obj = obj.__getattr__(s)

def get_kept_filter_channel(layers, layer_name, pr, kept_wg, wg='filter'):
    """Considering layer dependency, get the kept filters and channels for the layer of 'layer_name'.
    """
    current_layer = layers[layer_name]
    if wg in ["channel"]:
        kept_chl = kept_wg[layer_name]
        next_learnable = get_next_learnable(layers, layer_name)
        kept_filter = list(range(current_layer.module.weight.size(0))) if next_learnable is None else kept_wg[next_learnable]
    
    elif wg in ["filter"]:
        kept_filter = kept_wg[layer_name]
        prev_learnable = get_prev_learnable(layers, layer_name)
        if (prev_learnable is None) or pr[prev_learnable] == 0: 
            # In the case of SR networks, tail, there is an upsampling via sub-pixel. 'self.pr[prev_learnable_layer] == 0' can help avoid it. 
            # Not using this, the code will report error.
            kept_chl = list(range(current_layer.module.weight.size(1)))
        else:
            kept_chl = kept_wg[prev_learnable]
    return kept_filter, kept_chl

def get_masks(layers, pruned_wg):
    """Get masks for unstructured pruning.
    """
    masks = OrderedDict()
    for name, layer in layers.items():
        mask = torch.ones(layer.shape).cuda().flatten()
        mask[pruned_wg[name]] = 0
        masks[name] = mask.view(layer.shape)
    return masks